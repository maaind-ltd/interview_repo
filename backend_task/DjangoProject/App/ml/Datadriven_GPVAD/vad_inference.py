import torch
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from App.ml.Datadriven_GPVAD import utils
from django.conf import settings
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from App.ml.Datadriven_GPVAD.models import crnn

SAMPLE_RATE = 22050
EPS = np.spacing(1)
LMS_ARGS = {
    'n_fft': 2048,
    'n_mels': 64,
    'hop_length': int(SAMPLE_RATE * 0.02),
    'win_length': int(SAMPLE_RATE * 0.04)
}
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
DEVICE = torch.device(DEVICE)


def extract_feature(wavefilepath, **kwargs):
    wav, sr = sf.read(wavefilepath, dtype='float32')
    if wav.ndim > 1:
        wav = wav.mean(-1)
    wav = librosa.resample(wav, sr, target_sr=SAMPLE_RATE)
    return np.log(
        librosa.feature.melspectrogram(wav.astype(np.float32), sr, **kwargs) +
        EPS).T


class OnlineLogMelDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, **kwargs):
        super().__init__()
        self.dlist = data_list
        self.kwargs = kwargs

    def __getitem__(self, idx):
        return extract_feature(wavefilepath=self.dlist[idx],
                               **self.kwargs), self.dlist[idx]

    def __len__(self):
        return len(self.dlist)


MODELS = {
    't1': {
        'model': crnn,
        'outputdim': 527,
        'encoder': 'labelencoders/teacher.pth',
        'pretrained': 'teacher1/model.pth',
        'resolution': 0.02
    },
    't2': {
        'model': crnn,
        'outputdim': 527,
        'encoder': 'labelencoders/teacher.pth',
        'pretrained': 'teacher2/model.pth',
        'resolution': 0.02
    },
    'sre': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'sre/model.pth',
        'resolution': 0.02
    },
    'v2': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'vox2/model.pth',
        'resolution': 0.02
    },
    'a2': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'audioset2/model.pth',
        'resolution': 0.02
    },
    'a2_v2': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'audio2_vox2/model.pth',
        'resolution': 0.02
    },
    'c1': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'c1/model.pth',
        'resolution': 0.02
    },
}


# TODO: Have single parameter saying if it's hard or soft thresholding
def get_vad_prediction(wav_path='', speech_threshold=[0.5], hard=True, soft=False, 
                       vad_model_to_use='sre', model_pretrained_dir="pretrained_models"):
    pretrained_dir = Path(settings.VAD_MODEL_PATH)
    print("---------------------")
    print(pretrained_dir.resolve())
    print("---------------------")
    if not (pretrained_dir.exists() and pretrained_dir.is_dir()):
        logger.error(f"""Pretrained directory {pretrained_dir} not found.
        Please download the pretrained models from and try again or set --pretrained_dir to your
        directory.""")
        # TODO: Rather than return, maybe throw an exception?
        return
    wavlist = [wav_path]
    # print("wavlist = ", wavlist)
    dset = OnlineLogMelDataset(wavlist, **LMS_ARGS)
    dloader = torch.utils.data.DataLoader(dset,
                                          batch_size=1,
                                          num_workers=3,
                                          shuffle=False)

    model_kwargs_pack = MODELS[vad_model_to_use]
    # print('model_kwargs_pack', model_kwargs_pack)
    model_resolution = model_kwargs_pack['resolution']
    # Load model from relative path
    model = model_kwargs_pack['model'](
        outputdim=model_kwargs_pack['outputdim'],
        pretrained_from=pretrained_dir /
        model_kwargs_pack['pretrained']).to(DEVICE).eval()

    # print('model', model)
    encoder = torch.load(pretrained_dir / model_kwargs_pack['encoder'])
    logger.trace(model)

    output_dfs = []
    frame_outputs = {}
    threshold = tuple(speech_threshold)

    speech_label_idx = np.where('Speech' == encoder.classes_)[0].squeeze()
    # Using only binary thresholding without filter
    if len(threshold) == 1:
        postprocessing_method = utils.binarize
    else:
        postprocessing_method = utils.double_threshold
    with torch.no_grad(), tqdm(total=len(dloader), leave=False,
                               unit='clip') as pbar:
        for feature, filename in dloader:
            feature = torch.as_tensor(feature).to(DEVICE)
            prediction_tag, prediction_time = model(feature)
            prediction_tag = prediction_tag.to('cpu')
            prediction_time = prediction_time.to('cpu')

            # print("feature = ", feature.shape)

            if prediction_time is not None:  # Some models do not predict timestamps
                # print("Prediction time is NOT NONE")
                # print("hard = ", hard)
                cur_filename = filename[0]  # Remove batchsize
                thresholded_prediction = postprocessing_method(
                    prediction_time, *threshold)
                speech_soft_pred = prediction_time[..., speech_label_idx]
                if soft:
                    speech_soft_pred = prediction_time[
                        ..., speech_label_idx].numpy()
                    frame_outputs[cur_filename] = speech_soft_pred[
                        0]  # 1 batch

                if hard:
                    speech_hard_pred = thresholded_prediction[...,
                                                              speech_label_idx]
                    frame_outputs[cur_filename] = speech_hard_pred[
                        0]  # 1 batch
                # frame_outputs_hard.append(thresholded_prediction)

                labelled_predictions = utils.decode_with_timestamps(
                    encoder, thresholded_prediction)
                pred_label_df = pd.DataFrame(
                    labelled_predictions[0],
                    columns=['event_label', 'onset', 'offset'])

                # print("frame_outputs = ", frame_outputs)
                if not pred_label_df.empty:
                    pred_label_df['filename'] = cur_filename
                    pred_label_df['onset'] *= model_resolution
                    pred_label_df['offset'] *= model_resolution
                    pbar.set_postfix(labels=','.join(
                        np.unique(pred_label_df['event_label'].values)))
                    pbar.update()
                    output_dfs.append(pred_label_df)

    # print("output_dfs = ", output_dfs)
    # full_prediction_df = pd.concat(output_dfs).reset_index()
    # prediction_df = full_prediction_df[full_prediction_df['event_label'] == 'Speech']
    # np.set_printoptions(suppress=True, precision=2, linewidth=np.inf)
    # TODO: Rewrite this so no for loop, since it's a single file only
    for fname, output in frame_outputs.items():
        assert(len(output) > 0)
        proportion_speech = np.sum(output) / len(output)
        print("Proportion of speech = ", proportion_speech)
        # Return true if at least half the speech sample has speech
        if(proportion_speech < 0.25):
            return False, proportion_speech
        else:
            return True, proportion_speech
            # print(f"Output = {np.sum(output)}")
            # print(f"Length = {len(list(output))}")
    # else: #non-hard and non-soft prediction (='formatted' prediction)
        # print(prediction_df)


if __name__ == "__main__":
    get_vad_prediction()

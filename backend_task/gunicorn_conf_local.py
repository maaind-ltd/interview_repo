import multiprocessing
from logging.config import dictConfig

loglevel="debug"
accesslog="-"
errorlog="-"
access_log_format='%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
bind="0.0.0.0:6543"
workers=5
reload=True
timeout=120
ssl_version="TLSv1_2"
# Daemonize the Gunicorn process (detach & enter background)
daemon = False

preload_app = True

capture_output=True

LOGFILE_INTERVAL_UNIT='midnight'
LOGFILE_INTERVAL_COUNT=1

logconfig_dict={
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
        },
        "file.gunicorn.access": {
            "level": "DEBUG",
            "class": "logging.handlers.SysLogHandler",
            'facility': 'local1',
            'address': '/dev/log',
            "formatter": "generic",
        },
        "file.gunicorn.error": {
            "level": "DEBUG",
            "class": "logging.handlers.SysLogHandler",
            'facility': 'local1',
            'address': '/dev/log',
            "formatter": "generic",
        },
    },
    "loggers": {
        "gunicorn.access" : {
            "level": "DEBUG",
            "handlers": ["console", "file.gunicorn.access"],
            "propagate": False,
        },
        "gunicorn.error" : {
            "level": "DEBUG",
            "handlers": ["console", "file.gunicorn.error"],
            "propagate": False,
        },
    },
    "formatters": {
        "generic": {
            "format": "GUNICORN %(asctime)s [%(process)d] [%(levelname)s] %(message)s",
            "datefmt": "[%Y-%m-%d %H:%M:%S %z]",
            "class": "logging.Formatter"
        }
    }
}

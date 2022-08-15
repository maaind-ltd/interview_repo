from rest_framework import views
from rest_framework import status
from rest_framework.response import Response

# for the health checks                                                                                                     
class Index(views.APIView):
    def get(self, request):
        return Response(True, status=status.HTTP_200_OK)

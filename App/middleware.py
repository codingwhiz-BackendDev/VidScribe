# middleware.py
import os

class TempVideoCleanupMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        
        # Clean up temporary video file after response
        if 'temp_video_to_delete' in request.session:
            try:
                temp_file = request.session.pop('temp_video_to_delete')
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"Removed temporary video: {temp_file}")
            except Exception as e:
                print(f"Error removing temporary video: {e}")
        
        return response
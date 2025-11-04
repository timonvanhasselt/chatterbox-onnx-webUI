import os
import sys
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import cgi # Needed for parsing multipart/form-data
import uuid # Needed for unique file names
import shutil # NEW: Needed for directory cleanup
import atexit # NEW: Needed to register cleanup function

# Add the directory to the path and import the TTS class
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from chatterbox_onnx import ChatterboxOnnx
except ImportError as e:
    print(f"Error: Could not import 'ChatterboxOnnx': {e}")
    sys.exit(1)


# --- Configuration ---
EXAGGERATION_VALUE = 0.6
TARGET_VOICE = None
UPLOAD_DIR = "uploaded_voices" # Directory to store uploaded audio
PORT = 8002

# Initialize the synthesizer globally
try:
    print("INFO: Initializing ChatterboxOnnx (this may take a moment)...")
    synthesizer = ChatterboxOnnx(quantized=True)
except Exception as e:
    print(f"FATAL ERROR: Could not initialize the synthesizer: {e}")
    synthesizer = None


# NEW: Function to empty the uploaded voices directory when the script stops
def cleanup_uploaded_voices():
    if os.path.exists(UPLOAD_DIR):
        try:
            # We empty the directory, but do not remove it (safer than rmtree)
            for filename in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"WARNING: Could not remove file/directory '{file_path}' on exit: {e}")
            print(f"INFO: The upload directory '{UPLOAD_DIR}' successfully emptied on exit.")
        except OSError as e:
            print(f"WARNING: Could not empty the upload directory '{UPLOAD_DIR}' on exit: {e}")


class SimpleTTSRequestHandler(BaseHTTPRequestHandler):
    """
    Handler for the simple Python HTTP server.
    """
    
    def _serve_file(self, path, content_type):
        """Helper function to serve local files (such as index.html)."""
        try:
            with open(path, 'rb') as f:
                self.send_response(200)
                self.send_header('Content-type', content_type)
                self.end_headers()
                self.wfile.write(f.read())
        except FileNotFoundError:
            self.send_error(404, f"{path} not found.")

    def do_GET(self):
        """Handle GET requests (only for index.html)."""
        if self.path == '/':
            self._serve_file('index.html', 'text/html')
        else:
            self.send_error(404, "File not found.")

    def do_POST(self):
        """Handle POST requests (for /synthesize and /upload_voice)."""
        if synthesizer is None:
            self.send_error(503, "Synthesizer is not loaded.")
            return

        if self.path == '/synthesize':
            # --- Handling TTS synthesis ---
            try:
                # 1. Read the length of the request body
                content_length = int(self.headers['Content-Length'])
                # 2. Read the request body (JSON)
                post_data = self.rfile.read(content_length)
                
                # 3. Parse the JSON
                data = json.loads(post_data)
                text_to_speak = data.get('text', '').strip()
                voice_path = data.get('target_voice_path', TARGET_VOICE)
                language_id = data.get('language_id', 'nl')
            
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                self.send_error(400, f"Invalid request or JSON format: {e}")
                return

            if not text_to_speak:
                self.send_error(400, "No text provided.")
                return
            
            print(f"\n[SERVER] TTS request for: '{text_to_speak[:40]}...' (Language: {language_id}, Voice: {voice_path or 'Default'})")

            try:
                # 4. Generate audio to BytesIO buffer
                audio_buffer = synthesizer.synthesize_to_bytesio(
                    text=text_to_speak,
                    language_id=language_id,
                    target_voice_path=voice_path,
                    exaggeration=EXAGGERATION_VALUE,
                )
                
                audio_bytes = audio_buffer.read()
                
                # *** IMPORTANT: No os.remove(voice_path) added here. ***
                
                # 5. Send the response headers
                self.send_response(200)
                self.send_header('Content-type', 'audio/wav')
                self.send_header('Content-Length', str(len(audio_bytes)))
                self.end_headers()
                
                # 6. Send the binary audio data
                self.wfile.write(audio_bytes)
                
            except Exception as e:
                error_message = f"Error during synthesis: {e}"
                print(f"❌ {error_message}", file=sys.stderr)
                self.send_error(500, error_message)
                
        elif self.path == '/upload_voice':
            # --- Handling file upload (reference audio) ---
            try:
                # Use cgi.FieldStorage to parse multipart/form-data
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={'REQUEST_METHOD': 'POST',
                             'CONTENT_TYPE': self.headers['Content-Type'],
                             })
                
                if 'audioFile' not in form:
                    self.send_error(400, "No file received under 'audioFile'. Field missing.")
                    return
                
                file_item = form['audioFile']

                if not file_item.file:
                    self.send_error(400, "No valid file in the 'audioFile' field.")
                    return

                if not os.path.exists(UPLOAD_DIR):
                    os.makedirs(UPLOAD_DIR)

                # Determine the filename and path
                filename = file_item.filename if file_item.filename else "uploaded_audio.wav"
                
                # Add a unique ID to prevent conflicts
                unique_filename = f"{uuid.uuid4()}_{filename}"
                file_path = os.path.join(UPLOAD_DIR, unique_filename)

                # Save the file
                with open(file_path, 'wb') as f:
                    f.write(file_item.file.read())
                
                print(f"\n[SERVER] File saved (made ready for use) as: {file_path}")

                # Send the response back to the client
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                # Important: The client expects the path back in 'target_voice_path'
                response_data = json.dumps({'target_voice_path': file_path})
                self.wfile.write(response_data.encode('utf-8'))

            except Exception as e:
                error_message = f"Error during file upload: {e}"
                print(f"❌ {error_message}", file=sys.stderr)
                # Send a 500 error in case of an issue
                self.send_error(500, error_message)
                
        else:
            self.send_error(404, "Unknown POST path.")


def run_simple_server():
    # Check and create the upload directory at startup
    if not os.path.exists(UPLOAD_DIR):
        try:
            os.makedirs(UPLOAD_DIR)
            print(f"INFO: Upload directory '{UPLOAD_DIR}' created.")
        except Exception as e:
            print(f"WARNING: Could not create upload directory: {e}")
            
    # NEW: Register the cleanup function to be executed upon exit
    atexit.register(cleanup_uploaded_voices)
            
    server_address = ('0.0.0.0', PORT)
    httpd = HTTPServer(server_address, SimpleTTSRequestHandler)
    print(f"--- Starting the Chatterbox TTS HTTP Server at http://127.0.0.1:{PORT}/ ---")
    print("Press Ctrl+C to stop. Uploaded files will then be cleaned up.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped. Uploaded files are being cleaned up...")

if __name__ == '__main__':
    if not os.path.exists("index.html"):
        print("❌ ERROR: 'index.html' not found.")
        sys.exit(1)
        
    run_simple_server()

import os
import sys
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import cgi # Nodig voor het parsen van multipart/form-data
import uuid # Nodig voor unieke bestandsnamen
import shutil # NIEUW: Nodig voor directory cleanup
import atexit # NIEUW: Nodig om cleanup functie te registreren

# Voeg de map toe aan het pad en importeer de TTS-klasse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from chatterbox_onnx import ChatterboxOnnx
except ImportError as e:
    print(f"Fout: Kan 'ChatterboxOnnx' niet importeren: {e}")
    sys.exit(1)


# --- Configuratie ---
EXAGGERATION_VALUE = 0.6 
TARGET_VOICE = None 
UPLOAD_DIR = "uploaded_voices" # Map om geüploade audio op te slaan
PORT = 8002

# Initialiseer de synthesizer globaal
try:
    print("INFO: Initialiseer ChatterboxOnnx (dit kan even duren)...")
    synthesizer = ChatterboxOnnx(quantized=True)
except Exception as e:
    print(f"FATALE FOUT: Kan de synthesizer niet initialiseren: {e}")
    synthesizer = None


# NIEUW: Functie om de geüploade stemmenmap bij het stoppen van het script te legen
def cleanup_uploaded_voices():
    if os.path.exists(UPLOAD_DIR):
        try:
            # We legen de map, maar verwijderen hem niet (veiliger dan rmtree)
            for filename in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"WAARSCHUWING: Kan bestand/map '{file_path}' niet verwijderen bij afsluiten: {e}")
            print(f"INFO: De uploadmap '{UPLOAD_DIR}' is succesvol geleegd bij afsluiten.")
        except OSError as e:
            print(f"WAARSCHUWING: Kan de uploadmap '{UPLOAD_DIR}' niet legen bij afsluiten: {e}")


class SimpleTTSRequestHandler(BaseHTTPRequestHandler):
    """
    Handler voor de simpele Python HTTP-server.
    """
    
    def _serve_file(self, path, content_type):
        """Hulpfunctie om lokale bestanden (zoals index.html) te serveren."""
        try:
            with open(path, 'rb') as f:
                self.send_response(200)
                self.send_header('Content-type', content_type)
                self.end_headers()
                self.wfile.write(f.read())
        except FileNotFoundError:
            self.send_error(404, f"{path} niet gevonden.")

    def do_GET(self):
        """Afhandeling van GET-verzoeken (alleen voor index.html)."""
        if self.path == '/':
            self._serve_file('index.html', 'text/html')
        else:
            self.send_error(404, "Bestand niet gevonden.")

    def do_POST(self):
        """Afhandeling van POST-verzoeken (voor /synthesize en /upload_voice)."""
        if synthesizer is None:
            self.send_error(503, "Synthesizer is niet geladen.")
            return

        if self.path == '/synthesize':
            # --- Afhandeling van TTS-synthese ---
            try:
                # 1. Lees de lengte van de request body
                content_length = int(self.headers['Content-Length'])
                # 2. Lees de request body (JSON)
                post_data = self.rfile.read(content_length)
                
                # 3. Parse de JSON
                data = json.loads(post_data)
                text_to_speak = data.get('text', '').strip()
                voice_path = data.get('target_voice_path', TARGET_VOICE) 
                language_id = data.get('language_id', 'nl') 
            
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                self.send_error(400, f"Ongeldig verzoek of JSON-formaat: {e}")
                return

            if not text_to_speak:
                self.send_error(400, "Geen tekst opgegeven.")
                return
            
            print(f"\n[SERVER] TTS-aanvraag voor: '{text_to_speak[:40]}...' (Taal: {language_id}, Stem: {voice_path or 'Standaard'})")

            try:
                # 4. Genereer audio naar BytesIO buffer
                audio_buffer = synthesizer.synthesize_to_bytesio(
                    text=text_to_speak,
                    language_id=language_id,  
                    target_voice_path=voice_path,  
                    exaggeration=EXAGGERATION_VALUE,
                )
                
                audio_bytes = audio_buffer.read()
                
                # *** BELANGRIJK: Hier is geen os.remove(voice_path) toegevoegd. ***
                
                # 5. Stuur de response headers
                self.send_response(200)
                self.send_header('Content-type', 'audio/wav')
                self.send_header('Content-Length', str(len(audio_bytes)))
                self.end_headers()
                
                # 6. Stuur de binaire audio data
                self.wfile.write(audio_bytes)
                
            except Exception as e:
                error_message = f"Fout tijdens synthese: {e}"
                print(f"❌ {error_message}", file=sys.stderr)
                self.send_error(500, error_message)
                
        elif self.path == '/upload_voice':
            # --- Afhandeling van bestandsupload (referentie-audio) ---
            try:
                # Gebruik cgi.FieldStorage om multipart/form-data te parsen
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={'REQUEST_METHOD': 'POST',
                             'CONTENT_TYPE': self.headers['Content-Type'],
                             })
                
                if 'audioFile' not in form:
                    self.send_error(400, "Geen bestand ontvangen onder 'audioFile'. Veld ontbreekt.")
                    return
                
                file_item = form['audioFile']

                if not file_item.file:
                    self.send_error(400, "Geen geldig bestand in het 'audioFile' veld.")
                    return

                if not os.path.exists(UPLOAD_DIR):
                    os.makedirs(UPLOAD_DIR)

                # Bepaal de bestandsnaam en het pad
                filename = file_item.filename if file_item.filename else "uploaded_audio.wav"
                
                # Voeg een unieke ID toe om conflicten te voorkomen
                unique_filename = f"{uuid.uuid4()}_{filename}" 
                file_path = os.path.join(UPLOAD_DIR, unique_filename)

                # Sla het bestand op
                with open(file_path, 'wb') as f:
                    f.write(file_item.file.read())
                
                print(f"\n[SERVER] Bestand opgeslagen (gebruiksklaar gemaakt) als: {file_path}")

                # Stuur de response terug naar de client
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                # Belangrijk: De client verwacht het pad terug in 'target_voice_path'
                response_data = json.dumps({'target_voice_path': file_path})
                self.wfile.write(response_data.encode('utf-8'))

            except Exception as e:
                error_message = f"Fout tijdens bestandsupload: {e}"
                print(f"❌ {error_message}", file=sys.stderr)
                # Stuur een 500-fout in geval van een probleem
                self.send_error(500, error_message)
                
        else:
            self.send_error(404, "Onbekend POST-pad.")


def run_simple_server():
    # Controleer en maak de uploadmap bij de start
    if not os.path.exists(UPLOAD_DIR):
        try:
            os.makedirs(UPLOAD_DIR)
            print(f"INFO: Uploadmap '{UPLOAD_DIR}' aangemaakt.")
        except Exception as e:
            print(f"WAARSCHUWING: Kan uploadmap niet aanmaken: {e}")
            
    # NIEUW: Registreer de cleanup functie om uit te voeren bij afsluiten
    atexit.register(cleanup_uploaded_voices)
            
    server_address = ('0.0.0.0', PORT)
    httpd = HTTPServer(server_address, SimpleTTSRequestHandler)
    print(f"--- Start de Chatterbox TTS HTTP Server op http://127.0.0.1:{PORT}/ ---")
    print("Druk op Ctrl+C om te stoppen. Uploadbestanden worden dan opgeschoond.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer gestopt. Uploadbestanden worden opgeschoond...")

if __name__ == '__main__':
    if not os.path.exists("index.html"):
        print("❌ FOUT: 'index.html' niet gevonden.")
        sys.exit(1)
        
    run_simple_server()
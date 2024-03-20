import firebase_admin
import datetime	
from firebase_admin import credentials, storage

# Initialize Firebase Admin SDK with the service account key and storage bucket name
cred = credentials.Certificate("config1.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'crowd-91e59.appspot.com'
})

# Access Firebase Storage
bucket = storage.bucket()
local_file_path = "crowd_output_1.mp4"
firebase_storage_path = "test9.mp4"
blob = bucket.blob(firebase_storage_path)
blob.upload_from_filename(local_file_path)
download_url = blob.public_url
#signed_url = bucket.blob(firebase_storage_path).generate_signed_url(expiration=None)


signed_url = bucket.blob(firebase_storage_path).generate_signed_url(expiration=datetime.timedelta(minutes=60))
print(signed_url)

print("File uploaded to Firebase Storage.")
print("Download URL:", download_url)


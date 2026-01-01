from datetime import timedelta
from io import BytesIO

from minio import Minio

from backend.config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

BUCKET_NAME = "forensic-images"


def ensure_bucket_exists():
    if not client.bucket_exists(BUCKET_NAME):
        client.make_bucket(BUCKET_NAME)


def extract_object_name(storage_url: str) -> str:
    """Extract object name from storage_url (handles both old URLs and new object names)."""
    prefix = f"http://{MINIO_ENDPOINT}/{BUCKET_NAME}/"
    if storage_url.startswith(prefix):
        return storage_url[len(prefix):]
    return storage_url


def upload_file(data: bytes, object_name: str, content_type: str) -> str:
    """Upload file bytes to MinIO and return the object name (not URL)."""
    ensure_bucket_exists()
    client.put_object(
        BUCKET_NAME,
        object_name,
        BytesIO(data),
        length=len(data),
        content_type=content_type
    )
    return object_name


def get_presigned_url(object_name: str, expires: timedelta = timedelta(hours=1)) -> str:
    """Generate a presigned URL for accessing an object."""
    clean_name = extract_object_name(object_name)
    return client.presigned_get_object(BUCKET_NAME, clean_name, expires=expires)


def delete_file(object_name: str) -> None:
    """Delete a file from MinIO."""
    clean_name = extract_object_name(object_name)
    client.remove_object(BUCKET_NAME, clean_name)


def get_file_bytes(object_name: str) -> bytes:
    """Retrieve a file's bytes from MinIO."""
    clean_name = extract_object_name(object_name)
    response = None
    try:
        response = client.get_object(BUCKET_NAME, clean_name)
        return response.read()
    finally:
        if response:
            response.close()
            
            
    response.release_conn()

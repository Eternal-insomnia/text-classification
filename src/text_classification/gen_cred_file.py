import os

from dotenv import load_dotenv

load_dotenv()

aws_access_key_id = os.getenv('RUSTFS_ACCESS_KEY', 'write id at RUSTFS_ACCESS_KEY at .env')
aws_secret_access_key = os.getenv('RUSTFS_SECRET_KEY', 'write id at RUSTFS_SECRET_KEY at .env')

file = ".dvc-aws-credentials"

# create local aws credentials for dvc
with open(file=file, mode="w", encoding="UTF-8") as f:
    f.write("[default]\n")
    f.write(f"aws_access_key_id={aws_access_key_id}\n")
    f.write(f"aws_secret_access_key={aws_secret_access_key}\n")

import boto3
import os
import hashlib


def download(bucket_name, prefix=None, local_download_directory='./downloads'):

    # Retrieve the list of existing buckets
    s3 = boto3.client('s3')

    os.makedirs(local_download_directory, exist_ok=True)

    log_file_hash = hashlib.md5(f"{bucket_name}_{prefix}".encode()).hexdigest()
    downloaded_files_log = f'./downloaded_files_{log_file_hash}.txt'
    # Ensure the log file exists
    if not os.path.exists(downloaded_files_log):
        with open(downloaded_files_log, 'w') as f:
            pass  # Create the file if it doesn't exist

    # Read the list of already downloaded files from the log
    with open(downloaded_files_log, 'r') as f:
        downloaded_files = set(f.read().splitlines())

    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    new_files_count = 0
    # Check if the response contains any files
    if 'Contents' in response:
        for obj in response['Contents']:
            s3_key = obj['Key']
            local_file_path = os.path.join(local_download_directory, s3_key)
            local_file_path = os.path.splitext(local_file_path)[0] + '.png'

            if s3_key not in downloaded_files:
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                print(f"Downloading {s3_key} to {local_file_path}")
                s3.download_file(bucket_name, s3_key, local_file_path)
                print(f"Downloaded {s3_key} successfully.")

                with open(downloaded_files_log, 'a') as log_file:
                    log_file.write(s3_key + '\n')

                new_files_count += 1
            else:
                print(f"File {s3_key} already downloaded, skipping.")
    else:
        print(f"No files found in {bucket_name}/{prefix}")

    print(f"Total new files downloaded: {new_files_count}")


def list_files_with_structure(bucket_name, prefix=None):
    # Create an S3 client
    s3 = boto3.client('s3')

    # List the objects within the specified bucket and prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    # Check if there are any contents
    if 'Contents' not in response:
        print(f"No files found in {bucket_name}/{prefix}")
        return

    # Organize files into a tree structure
    file_tree = {}

    for obj in response['Contents']:
        file_key = obj['Key']
        parts = file_key.split('/')

        # Traverse through the tree structure and build it up
        current_level = file_tree
        for part in parts[:-1]:  # Skip the file itself in the path
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]

        # The last part is the file name, not a directory
        current_level[parts[-1]] = None  # File node

    # Print the file structure
    def print_tree(tree, indent=''):
        for key, value in tree.items():
            if value is None:  # It's a file
                print(f"{indent}- {key}")
            else:  # It's a directory
                print(f"{indent}{key}/")
                print_tree(value, indent + '  ')

    print(f"Files and directories in {bucket_name}/{prefix}:")
    print_tree(file_tree)


if __name__ == "__main__":

    bucket_name = "one-app-develop-s3"  # 'one-app-prod-s3'  # 'one-app-develop-s3'
    prefix = "avatar-2d/input/"
    local_download_directory = os.path.join("./downloads", bucket_name)
    # list_files_with_structure(bucket_name, prefix)
    download(bucket_name, prefix, local_download_directory)

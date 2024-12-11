import zipfile
import os

def zip_folder(folder_path, output_path):
    """
    Zips the contents of a folder.

    :param folder_path: The path of the folder to zip.
    :param output_path: The path where the zip file will be saved.
    """
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
    print(f"Folder '{folder_path}' has been zipped into '{output_path}'.")

# Example usage
folder_to_zip = "/drive2/tuandung/WCODELLM/jaist/magic_coder"
output_zip_file = "/drive2/tuandung/WCODELLM/jaist/magic_coder.zip"
zip_folder(folder_to_zip, output_zip_file)
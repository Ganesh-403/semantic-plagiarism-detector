import zipfile
import io

def _validate_opendocument_archive(file_bytes: bytes, filename: str) -> bool:
    """
    Validates that an OpenDocument Text (.odt) archive has a valid ZIP structure
    where the first entry is named 'mimetype' containing 'application/vnd.oasis.opendocument.text'.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            infolist = zf.infolist()
            if not infolist:
                return False
                
            # The OpenDocument specification requires the 'mimetype' file 
            # to be the first file in the archive and uncompressed (compress_type == 0 / ZIP_STORED)
            first_member = infolist[0]
            if first_member.filename != "mimetype" or first_member.compress_type != zipfile.ZIP_STORED:
                return False
                
            # Read and verify content of the mimetype file
            with zf.open(first_member) as f:
                content = f.read().strip()
                if content != b"application/vnd.oasis.opendocument.text":
                    return False
                    
            # Optionally verify presence of content.xml for completeness
            if "content.xml" not in zf.namelist():
                return False
                
        return True
    except (zipfile.BadZipFile, Exception):
        return False
    

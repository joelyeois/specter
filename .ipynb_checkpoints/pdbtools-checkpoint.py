from urllib.request import urlopen
from urllib.error import HTTPError
from gzip import GzipFile
from io import BytesIO
from pathlib import Path
import os

def fetch_pdb_file(pdbcode, output='./', force=False, assembly=False):
    """
    Download a PDB file and save it in a given location.

    Parameters
    ----------
    pdbcode : str
        A valid PDB code
    output : str, optional
        The destination for the PDB file to be saved
    force : bool, optional
        Download PDB file even if it already exists
    assembly : bool, optional
        Download biological assembly

    Returns
    -------
    str
        Path to the saved PDB file 
    """
    url = "https://files.rcsb.org/download/{code}.pdb{assembly}.gz".format(code=pdbcode, assembly="1" if assembly else "")
    filename = Path(output) / "{}.pdb".format(pdbcode)  #"".join([output,'/',pdbcode, '.pdb'])
    if (os.path.isfile(filename)) and not force:
        return filename
    try:
        response = urlopen(url)
    except HTTPError:
        raise IOError("Error 404: {url} not found".format(url=url))
    compressed = BytesIO()
    compressed.write(response.read())
    compressed.seek(0)
    decompressed = GzipFile(fileobj=compressed, mode='rb')
    with open(filename, "wb") as f:
        f.write(decompressed.read())
    return filename
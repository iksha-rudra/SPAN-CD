# from pyunpack import Archive

# Archive('/home/rakesh/DataSet/S2MTCP/S2MTCP_data_2.7z').\
#     extractall('/home/rakesh/DataSet/S2MTCP/')
    
from py7zr import unpack_7zarchive
import shutil

shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)
shutil.unpack_archive('/home/rakesh/DataSet/S2MTCP/S2MTCP_data_2.7z', '/home/rakesh/DataSet/S2MTCP/')
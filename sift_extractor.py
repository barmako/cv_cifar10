import dataloader
import sift_extractor_utils as seu

pickle = dataloader.get_pickle(0)
data = pickle[0]
datum = data[0]

sift = seu.extract_sift(datum)
flat_sift = seu.concat_sifts(sift)
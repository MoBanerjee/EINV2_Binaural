
from methods.ein_seld.crop import Crop
from methods.ein_seld.freqshift import FreqShift
from methods.ein_seld.specaug import SpecAugment
from methods.ein_seld.trackmix import TrackMix
from methods.ein_seld.wavmix import WavMix


cropper=Crop()
shifter=FreqShift()
spec=SpecAugment()
trackmixer=TrackMix()
wavmixer=WavMix()
augmentations = [
cropper,shifter,spec,trackmixer,wavmixer
]
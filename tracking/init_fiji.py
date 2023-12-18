import imagej
import scyjava


def track(full_list, fiji_path):
    scyjava.config.add_option('-Xmx6g')
    ij = imagej.init(fiji_path)
    for volume in full_list:
        script = f"""
import sys
from ij import IJ
from ij import WindowManager

from fiji.plugin.trackmate import Model
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import SelectionModel
from fiji.plugin.trackmate import Logger
from fiji.plugin.trackmate.detection import MaskDetectorFactory
from fiji.plugin.trackmate.detection import LabelImageDetectorFactory
from fiji.plugin.trackmate.tracking.kalman import AdvancedKalmanTrackerFactory
from fiji.plugin.trackmate.io import TmXmlWriter
import fiji.plugin.trackmate.action.ExportTracksToXML as ExportTracksToXML
from java.io import File

# We have to do the following to avoid errors with UTF8 chars generated in
# TrackMate that will mess with our Fiji Jython.
reload(sys)
sys.setdefaultencoding('utf-8') 

# Get currently selected image
# imp = WindowManager.getCurrentImage()
path = '{str(volume).replace("segmentations", "trackings")}'
imp = IJ.openImage('{str(volume)}')
dims = imp.getDimensions()
imp.setDimensions( dims[ 2 ], dims[ 4 ], dims[ 3 ] )
print(imp.getDimensions())
model = Model()

# Send all messages to ImageJ log window.
model.setLogger(Logger.IJ_LOGGER)
settings = Settings(imp)

# Configure detector - We use the Strings for the keys
# settings.detectorFactory = MaskDetectorFactory()
settings.detectorFactory = LabelImageDetectorFactory()
settings.detectorSettings = {{
    'TARGET_CHANNEL': 1,
    'SIMPLIFY_CONTOURS': True,
}}
settings.trackerFactory = AdvancedKalmanTrackerFactory()
settings.trackerSettings = settings.trackerFactory.getDefaultSettings()  # almost good enough
settings.trackerSettings['ALLOW_TRACK_SPLITTING'] = False
settings.trackerSettings['ALLOW_TRACK_MERGING'] = False
# settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = 30.0
settings.trackerSettings['SPLITTING_MAX_DISTANCE'] = 30.0
settings.trackerSettings['MERGING_MAX_DISTANCE'] = 30.0
settings.trackerSettings['LINKING_MAX_DISTANCE'] = 50.0
settings.trackerSettings['KALMAN_SEARCH_RADIUS'] = 100.0
settings.trackerSettings['MAX_FRAME_GAP'] = 2
settings.addAllAnalyzers()
trackmate = TrackMate(model, settings)
ok = trackmate.checkInput()
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))

ok = trackmate.process()
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))

filename = '{str(volume).replace('tif', '').replace("segmentations", "trackings")}'
outFile = File(filename + "exportModel.xml")  # this will write the full trackmate xml.
writer = TmXmlWriter(outFile)
writer.appendModel(model)
writer.appendSettings(settings)
writer.writeToFile()
        """
        res = ij.py.run_script(language="py", script=script)
        print(f"Tracking done for {volume.name}!")
    ij.dispose()
    #scyjava.jimport('java.lang.System').exit(0)

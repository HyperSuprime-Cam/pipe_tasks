import numpy

from lsst.pipe.base import CmdLineTask, Struct, TaskRunner, ArgumentParser, ButlerInitializedTaskRunner
from lsst.pex.config import Config, Field, ListField, ConfigurableField, RangeField, ConfigField
from lsst.meas.algorithms import SourceDetectionTask, SourceMeasurementTask
from lsst.meas.deblender import SourceDeblendTask
from lsst.pipe.tasks.coaddBase import getSkyInfo, ExistingCoaddDataIdContainer, scaleVariance
from lsst.pipe.tasks.astrometry import AstrometryTask
from lsst.pipe.tasks.setPrimaryFlags import SetPrimaryFlagsTask
from lsst.pipe.tasks.propagateVisitFlags import PropagateVisitFlagsTask
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath
import lsst.afw.detection as afwDetect
from lsst.daf.base import PropertyList

"""
New dataset types:
* deepCoadd_det: detections from what used to be processCoadd (tract, patch, filter)
* deepCoadd_mergeDet: merged detections (tract, patch)
* deepCoadd_meas: measurements of merged detections (tract, patch, filter)
* deepCoadd_ref: reference sources (tract, patch)
All of these have associated *_schema catalogs that require no data ID and hold no records.

In addition, we have a schema-only dataset, which saves the schema for the PeakRecords in
the mergeDet, meas, and ref dataset Footprints:
* deepCoadd_peak_schema
"""


def _makeGetSchemaCatalogs(datasetSuffix):
    """Construct a getSchemaCatalogs instance method

    These are identical for most of the classes here, so we'll consolidate
    the code.

    datasetSuffix:  Suffix of dataset name, e.g., "src" for "deepCoadd_src"
    """
    def getSchemaCatalogs(self):
        """Return a dict of empty catalogs for each catalog dataset produced by this task."""
        src = afwTable.SourceCatalog(self.schema)
        if hasattr(self, "algMetadata"):
            src.getTable().setMetadata(self.algMetadata)
        return {self.config.coaddName + "Coadd_" + datasetSuffix: src}
    return getSchemaCatalogs

def _makeMakeIdFactory(datasetName):
    """Construct a makeIdFactory instance method

    These are identical for all the classes here, so this consolidates
    the code.

    datasetName:  Dataset name without the coadd name prefix, e.g., "CoaddId" for "deepCoaddId"
    """
    def makeIdFactory(self, dataRef):
        """Return an IdFactory for setting the detection identifiers

        The actual parameters used in the IdFactory are provided by
        the butler (through the provided data reference.
        """
        expBits = dataRef.get(self.config.coaddName + datasetName + "_bits")
        expId = long(dataRef.get(self.config.coaddName + datasetName))
        return afwTable.IdFactory.makeSource(expId, 64 - expBits)
    return makeIdFactory


def copySlots(oldCat, newCat):
    """Copy table slots definitions from one catalog to another"""
    for name in ("Centroid", "Shape", "ApFlux", "ModelFlux", "PsfFlux", "InstFlux", "CalibFlux"):
        meas = getattr(oldCat.table, "get" + name + "Key")()
        err = getattr(oldCat.table, "get" + name + "ErrKey")()
        flag = getattr(oldCat.table, "get" + name + "FlagKey")()
        getattr(newCat.table, "define" + name)(meas, err, flag)


def getShortFilterName(name):
    """Given a longer, camera-specific filter name (e.g. "HSC-I") return its shorthand name ("i").
    """
    # I'm not sure if this is the way this is supposed to be implemented, but it seems to work,
    # and its the only way I could get it to work.
    return afwImage.Filter(name).getFilterProperty().getName()


##############################################################################################################

class DetectCoaddSourcesConfig(Config):
    doScaleVariance = Field(dtype=bool, default=True, doc="Scale variance plane using empirical noise?")
    detection = ConfigurableField(target=SourceDetectionTask, doc="Source detection")
    coaddName = Field(dtype=str, default="deep", doc="Name of coadd")

    def setDefaults(self):
        Config.setDefaults(self)
        self.detection.thresholdType = "pixel_stdev"
        self.detection.isotropicGrow = True
        # Coadds are made from background-subtracted CCDs, so background subtraction should be very basic
        self.detection.background.useApprox = False
        self.detection.background.binSize = 4096
        self.detection.background.undersampleStyle = 'REDUCE_INTERP_ORDER'


class DetectCoaddSourcesTask(CmdLineTask):
    """Detect sources on a coadd

    This operation is performed separately in each band.  The detections from
    each band will be merged before performing the measurement stage.
    """
    _DefaultName = "detectCoaddSources"
    ConfigClass = DetectCoaddSourcesConfig
    getSchemaCatalogs = _makeGetSchemaCatalogs("det")
    makeIdFactory = _makeMakeIdFactory("CoaddId")

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "deepCoadd", help="data ID, e.g. --id tract=12345 patch=1,2 filter=r",
                               ContainerClass=ExistingCoaddDataIdContainer)
        return parser

    def __init__(self, schema=None, **kwargs):
        """Initialize the task.

        Keyword arguments (in addition to those forwarded to CmdLineTask.__init__):
         - schema: the initial schema for the output catalog, modified-in place to include all
                   fields set by this task.  If None, the source minimal schema will be used.
        """
        CmdLineTask.__init__(self, **kwargs)
        if schema is None:
            schema = afwTable.SourceTable.makeMinimalSchema()
        self.schema = schema
        self.makeSubtask("detection", schema=self.schema)

    def run(self, patchRef):
        exposure = patchRef.get(self.config.coaddName + "Coadd", immediate=True)
        results = self.runDetection(exposure, self.makeIdFactory(patchRef))
        self.write(results, patchRef)
        patchRef.put(exposure, self.config.coaddName + "Coadd_calexp")
        return results

    def runDetection(self, exposure, idFactory):
        """Run detection on an exposure

        exposure: Exposure on which to detect
        idFactory: IdFactory to set source identifiers

        Returns: Struct(sources: catalog of detections,
                        backgrounds: list of backgrounds
                        )
        """
        if self.config.doScaleVariance:
            scaleVariance(exposure.getMaskedImage(), log=self.log)
        backgrounds = afwMath.BackgroundList()
        table = afwTable.SourceTable.make(self.schema, idFactory)
        detections = self.detection.makeSourceCatalog(table, exposure)
        sources = detections.sources
        fpSets = detections.fpSets
        if fpSets.background:
            backgrounds.append(fpSets.background)
        return Struct(sources=sources, backgrounds=backgrounds)

    def write(self, results, patchRef):
        """Write out results from runDetection

        results: Struct returned from runDetection
        patchRef: data reference for patch
        """
        coaddName = self.config.coaddName + "Coadd"
        patchRef.put(results.backgrounds, coaddName + "_calexpBackground")
        patchRef.put(results.sources, coaddName + "_det")


##############################################################################################################

class MergeSourcesRunner(TaskRunner):
    def makeTask(self, parsedCmd=None, args=None):
        """Provide a butler to the Task constructor"""
        if parsedCmd is not None:
            butler = parsedCmd.butler
        elif args is not None:
            dataRefList, kwargs = args
            butler = dataRefList[0].getButler()
        else:
            raise RuntimeError("Neither parsedCmd or args specified")
        return self.TaskClass(config=self.config, log=self.log, butler=butler)

    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        """Provide a list of patch references for each patch

        The patch references within the list will have different filters.
        """
        refList = {} # Will index this as refList[tract][patch][filter] = ref
        for ref in parsedCmd.id.refList:
            tract = ref.dataId["tract"]
            patch = ref.dataId["patch"]
            filter = ref.dataId["filter"]
            if not tract in refList:
                refList[tract] = {}
            if not patch in refList[tract]:
                refList[tract][patch] = {}
            if filter in refList[tract][patch]:
                raise RuntimeError("Multiple versions of %s" % (ref.dataId,))
            refList[tract][patch][filter] = ref
        return [(p.values(), kwargs) for t in refList.itervalues() for p in t.itervalues()]


class MergeSourcesConfig(Config):
    priorityList = ListField(dtype=str, default=[],
                             doc="Priority-ordered list of bands for the merge.")
    coaddName = Field(dtype=str, default="deep", doc="Name of coadd")

    def validate(self):
        Config.validate(self)
        if len(self.priorityList) == 0:
            raise RuntimeError("No priority list provided")


class MergeSourcesTask(CmdLineTask):
    """A base class for merging source catalogs

    Merging detections (MergeDetectionsTask) and merging measurements
    (MergeMeasurementsTask) are currently so similar that it makes
    sense to re-use the code, in the form of this abstract base class.

    Sub-classes should set the following class variables:
    * _DefaultName: name of Task
    * inputDataset: name of dataset to read
    * outputDataset: name of dataset to write
    * getSchemaCatalogs to the output of _makeGetSchemaCatalogs(outputDataset)

    In addition, sub-classes must implement the mergeCatalogs method.
    """
    _DefaultName = None
    ConfigClass = MergeSourcesConfig
    RunnerClass = MergeSourcesRunner
    inputDataset = None
    outputDataset = None
    getSchemaCatalogs = None

    @classmethod
    def _makeArgumentParser(cls):
        """Create a suitable ArgumentParser

        We will use the ArgumentParser to get a provide a list of data
        references for patches; the RunnerClass will sort them into lists
        of data references for the same patch
        """
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "deepCoadd_" + cls.inputDataset,
                               ContainerClass=ExistingCoaddDataIdContainer,
                               help="data ID, e.g. --id tract=12345 patch=1,2 filter=g^r^i")
        return parser

    def getInputSchema(self, butler=None, schema=None):
        if schema is None:
            assert butler is not None, "Neither butler nor schema specified"
            schema = butler.get(self.config.coaddName + "Coadd_" + self.inputDataset + "_schema",
                                immediate=True).schema
        return schema

    def __init__(self, butler=None, schema=None, **kwargs):
        """Initialize the task.

        Keyword arguments (in addition to those forwarded to CmdLineTask.__init__):
         - schema: the schema of the detection catalogs used as input to this one
         - butler: a butler used to read the input schema from disk, if schema is None

        Derived classes should use the getInputSchema() method to handle the additional
        arguments and retreive the actual input schema.
        """
        CmdLineTask.__init__(self, **kwargs)

    def run(self, patchRefList):
        """Merge coadd sources from multiple bands

        patchRefList: list of patch data reference for each filter
        """
        catalogs = dict(self.readCatalog(patchRef) for patchRef in patchRefList)
        mergedCatalog = self.mergeCatalogs(catalogs, patchRefList[0])
        self.write(patchRefList[0], mergedCatalog)

    def readCatalog(self, patchRef):
        """Read input catalog

        We read the input dataset provided by the 'inputDataset'
        class variable.
        """
        filterName = patchRef.dataId["filter"]
        catalog = patchRef.get(self.config.coaddName + "Coadd_" + self.inputDataset, immediate=True)
        self.log.info("Read %d sources for filter %s: %s" % (len(catalog), filterName, patchRef.dataId))
        return filterName, catalog

    def mergeCatalogs(self, catalogs, patchRef):
        """Merge multiple catalogs

        catalogs: dict mapping filter name to source catalog

        Returns: merged catalog
        """
        raise NotImplementedError()

    def write(self, patchRef, catalog):
        """Write the output

        We write as the dataset provided by the 'outputDataset'
        class variable.
        """
        patchRef.put(catalog, self.config.coaddName + "Coadd_" + self.outputDataset)
        # since the filter isn't actually part of the data ID for the dataset we're saving,
        # it's confusing to see it in the log message, even if the butler simply ignores it.
        mergeDataId = patchRef.dataId.copy()
        del mergeDataId["filter"]
        self.log.info("Wrote merged catalog: %s" % (mergeDataId,))

    def writeMetadata(self, dataRefList):
        """No metadata to write, and not sure how to write it for a list of dataRefs"""
        pass


class CullPeaksConfig(Config):
    """Configuration for culling garbage peaks after merging Footprints.

    Peaks may also be culled after detection or during deblending; this configuration object
    only deals with culling after merging Footprints.

    These cuts are based on three quantities:
     - nBands: the number of bands in which the peak was detected
     - peakRank: the position of the peak within its family, sorted from brightest to faintest.
     - peakRankNormalized: the peak rank divided by the total number of peaks in the family.

    The formula that identifie peaks to cull is:

      nBands < nBandsSufficient
        AND (rank >= rankSufficient)
        AND (rank >= rankConsider OR rank >= rankNormalizedConsider)

    To disable peak culling, simply set nBandsSafe=1.
    """

    nBandsSufficient = RangeField(dtype=int, default=2, min=1,
                                  doc="Always keep peaks detected in this many bands")
    rankSufficient = RangeField(dtype=int, default=20, min=1,
                                doc="Always keep this many peaks in each family")
    rankConsidered = RangeField(dtype=int, default=30, min=1,
                                doc=("Keep peaks with less than this rank that also match the "
                                     "rankNormalizedConsidered condition."))
    rankNormalizedConsidered = RangeField(dtype=float, default=0.7, min=0.0,
                                          doc=("Keep peaks with less than this normalized rank that"
                                               " also match the rankConsidered condition."))


class MergeDetectionsConfig(MergeSourcesConfig):
    minNewPeak = Field(dtype=float, default=1,
                       doc="Minimum distance from closest peak to create a new one (in arcsec).")

    maxSamePeak = Field(dtype=float, default=0.3,
                        doc="When adding new catalogs to the merge, all peaks less than this distance "
                        " (in arcsec) to an existing peak will be flagged as detected in that catalog.")
    cullPeaks = ConfigField(dtype=CullPeaksConfig, doc="Configuration for how to cull peaks.")

    skyFilterName = Field(dtype=str, default="sky",
                          doc="Name of `filter' used to label sky objects (e.g. flag merge.peak.sky is set)\n"
                          "(N.b. should be in MergeMeasurementsConfig.pseudoFilterList)")
    skySourceRadius = Field(dtype=float, default=8,
                            doc="Radius, in pixels, of sky objects")
    nSkySourcesPerPatch = Field(dtype=int, default=0,
                                doc="Try to add this many sky objects to the mergeDet list, which will\n"
                                "then be measured along with the detected objects in sourceMeasurementTask")
    nTrialSkySourcesPerPatch = Field(dtype=int, default=None, optional=True,
                                doc="Maximum number of trial sky object positions\n"
                                     "(default: nSkySourcesPerPatch*nTrialSkySourcesPerPatchMultiplier)")
    nTrialSkySourcesPerPatchMultiplier = Field(dtype=int, default=5,
                                               doc="Set nTrialSkySourcesPerPatch to\n"
                                               "    nSkySourcesPerPatch*nTrialSkySourcesPerPatchMultiplier\n"
                                               "if nTrialSkySourcesPerPatch is None")

class MergeDetectionsTask(MergeSourcesTask):
    """Merge detections from multiple bands"""
    ConfigClass = MergeDetectionsConfig
    _DefaultName = "mergeCoaddDetections"
    inputDataset = "det"
    outputDataset = "mergeDet"
    makeIdFactory = _makeMakeIdFactory("MergedCoaddId")

    def __init__(self, butler=None, schema=None, **kwargs):
        """Initialize the task.

        Additional keyword arguments (forwarded to MergeSourcesTask.__init__):
         - schema: the schema of the detection catalogs used as input to this one
         - butler: a butler used to read the input schema from disk, if schema is None

        The task will set its own self.schema attribute to the schema of the output merged catalog.
        """
        MergeSourcesTask.__init__(self, butler=butler, schema=schema, **kwargs)
        self.schema = self.getInputSchema(butler=butler, schema=schema)

        filterNames = [getShortFilterName(name) for name in self.config.priorityList]
        if self.config.nSkySourcesPerPatch > 0:
            filterNames += [self.config.skyFilterName]
        self.merged = afwDetect.FootprintMergeList(self.schema, filterNames)

    def mergeCatalogs(self, catalogs, patchRef):
        """Merge multiple catalogs
        """

        # Convert distance to tract coordiante
        skyInfo = getSkyInfo(coaddName=self.config.coaddName, patchRef=patchRef)
        tractWcs = skyInfo.wcs
        peakDistance = self.config.minNewPeak / tractWcs.pixelScale().asArcseconds()
        samePeakDistance = self.config.maxSamePeak / tractWcs.pixelScale().asArcseconds()

        # Put catalogs, filters in priority order
        orderedCatalogs = [catalogs[band] for band in self.config.priorityList if band in catalogs.keys()]
        orderedBands = [getShortFilterName(band) for band in self.config.priorityList
                        if band in catalogs.keys()]

        mergedList = self.merged.getMergedSourceCatalog(orderedCatalogs, orderedBands, peakDistance,
                                                        self.schema, self.makeIdFactory(patchRef),
                                                        samePeakDistance)
        copySlots(orderedCatalogs[0], mergedList)
        #
        # Add extra sources that correspond to blank sky
        #
        skySourceFootprints = self.getSkySourceFootprints(mergedList, skyInfo)
        if skySourceFootprints:
            key = mergedList.schema.find("merge.footprint.%s" % self.config.skyFilterName).key

            for foot in skySourceFootprints:
                s = mergedList.addNew()
                s.setFootprint(foot)
                s.set(key, True)

            self.log.info("Added %d sky sources (%.0f%% of requested)" % (
                len(skySourceFootprints), 100*len(skySourceFootprints)/float(self.config.nSkySourcesPerPatch)))

        # Sort Peaks from brightest to faintest
        for record in mergedList:
            record.getFootprint().sortPeaks()
        self.log.info("Merged to %d sources" % len(mergedList))
        # Attempt to remove garbage peaks
        self.cullPeaks(mergedList)
        return mergedList

    def cullPeaks(self, catalog):
        """Attempt to remove garbage peaks (mostly on the outskirts of large blends)"""
        keys = [item.key for item in self.merged.getPeakSchema().extract("merge.peak.*").itervalues()]
        totalPeaks = 0
        culledPeaks = 0
        for parentSource in catalog:
            # Make a list copy so we can clear the attached PeakCatalog and append the ones we're keeping
            # to it (which is easier than deleting as we iterate).
            keptPeaks = parentSource.getFootprint().getPeaks()
            oldPeaks = list(keptPeaks)
            keptPeaks.clear()
            familySize = len(oldPeaks)
            totalPeaks += familySize
            for rank, peak in enumerate(oldPeaks):
                if ((rank < self.config.cullPeaks.rankSufficient) or
                    (self.config.cullPeaks.nBandsSufficient > 1 and
                     sum([peak.get(k) for k in keys]) >= self.config.cullPeaks.nBandsSufficient) or
                    (rank < self.config.cullPeaks.rankConsidered and
                     rank < self.config.cullPeaks.rankNormalizedConsidered * familySize)):
                    keptPeaks.append(peak)
                else:
                    culledPeaks += 1
        self.log.info("Culled %d of %d peaks" % (culledPeaks, totalPeaks))

    def getSchemaCatalogs(self):
        """Return a dict of empty catalogs for each catalog dataset produced by this task."""
        mergeDet = afwTable.SourceCatalog(self.schema)
        peak = afwDetect.PeakCatalog(self.merged.getPeakSchema())
        return {self.config.coaddName + "Coadd_mergeDet": mergeDet,
                self.config.coaddName + "Coadd_peak": peak}

    def getSkySourceFootprints(self, mergedList, skyInfo):
        """!Return a list of Footprints of sky objects which don't overlap with anything in mergedList

        \param mergedList  The merged Footprints from all the input bands
        \param skyInfo     A description of the patch
        """

        if self.config.nSkySourcesPerPatch <= 0:
            return []

        skySourceRadius = self.config.skySourceRadius
        nSkySourcesPerPatch = self.config.nSkySourcesPerPatch
        nTrialSkySourcesPerPatch = self.config.nTrialSkySourcesPerPatch
        if nTrialSkySourcesPerPatch is None:
            nTrialSkySourcesPerPatch = self.config.nTrialSkySourcesPerPatchMultiplier*nSkySourcesPerPatch
        #
        # We are going to find circular Footprints that don't intersect any pre-existing Footprints,
        # and the easiest way to do this is to generate a Mask containing all the detected pixels (as
        # merged by this task).
        #
        patchBBox = skyInfo.patchInfo.getOuterBBox()
        mask = afwImage.MaskU(patchBBox)
        DETECTED = mask.getPlaneBitMask("DETECTED")
        for s in mergedList:
            afwDetect.setMaskFromFootprint(mask, s.getFootprint(), DETECTED)

        xmin, ymin = patchBBox.getMin()
        xmax, ymax = patchBBox.getMax()
        # Avoid objects partially off the image
        xmin += skySourceRadius + 1
        ymin += skySourceRadius + 1
        xmax -= skySourceRadius + 1
        ymax -= skySourceRadius + 1

        skySourceFootprints = []
        for i in range(nTrialSkySourcesPerPatch):
            if len(skySourceFootprints) == nSkySourcesPerPatch:
                break

            x = int(numpy.random.uniform(xmin, xmax))
            y = int(numpy.random.uniform(ymin, ymax))
            foot = afwDetect.Footprint(afwGeom.PointI(x, y), skySourceRadius, patchBBox)
            foot.setPeakSchema(self.merged.getPeakSchema())

            if not foot.overlapsMask(mask):
                foot.addPeak(x, y, 0)
                foot.getPeaks()[0].set("merge.peak.%s" % self.config.skyFilterName, True)
                skySourceFootprints.append(foot)

        return skySourceFootprints

##############################################################################################################

class MeasureMergedCoaddSourcesConfig(Config):
    doDeblend = Field(dtype=bool, default=True, doc="Deblend sources?")
    deblend = ConfigurableField(target=SourceDeblendTask, doc="Deblend sources")
    measurement = ConfigurableField(target=SourceMeasurementTask, doc="Source measurement")
    setPrimaryFlags = ConfigurableField(target=SetPrimaryFlagsTask, doc="Set flags for primary tract/patch")
    doPropagateFlags = Field(
        dtype=bool, default=True,
        doc="Whether to match sources to CCD catalogs to propagate flags (to e.g. identify PSF stars)"
    )
    propagateFlags = ConfigurableField(target=PropagateVisitFlagsTask, doc="Propagate visit flags to coadd")
    doMatchSources = Field(dtype=bool, default=True, doc="Match sources to reference catalog?")
    astrometry = ConfigurableField(target=AstrometryTask, doc="Astrometric matching")
    coaddName = Field(dtype=str, default="deep", doc="Name of coadd")

    def setDefaults(self):
        Config.setDefaults(self)
        self.deblend.propagateAllPeaks = True
        self.astrometry.forceKnownWcs=True
        self.astrometry.solver.calculateSip=False
        self.measurement.doBlendedness = True


class MeasureMergedCoaddSourcesTask(CmdLineTask):
    """Measure sources using the merged catalog of detections

    This operation is performed separately on each band.  We deblend and measure on
    the list of merge detections.  The results from each band will subsequently
    be merged to create a final reference catalog for forced measurement.
    """
    _DefaultName = "measureCoaddSources"
    ConfigClass = MeasureMergedCoaddSourcesConfig
    RunnerClass = ButlerInitializedTaskRunner
    getSchemaCatalogs = _makeGetSchemaCatalogs("meas")
    makeIdFactory = _makeMakeIdFactory("MergedCoaddId") # The IDs we already have are of this type

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "deepCoadd_calexp",
                               help="data ID, e.g. --id tract=12345 patch=1,2 filter=r",
                               ContainerClass=ExistingCoaddDataIdContainer)
        return parser

    def __init__(self, butler=None, schema=None, peakSchema=None, **kwargs):
        """Initialize the task.

        Keyword arguments (in addition to those forwarded to CmdLineTask.__init__):
         - schema: the schema of the merged detection catalog used as input to this one
         - peakSchema: the schema of the PeakRecords in the Footprints in the merged detection catalog
         - butler: a butler used to read the input schemas from disk, if schema or peakSchema is None

        The task will set its own self.schema attribute to the schema of the output measurement catalog.
        This will include all fields from the input schema, as well as additional fields for all the
        measurements.
        """
        CmdLineTask.__init__(self, **kwargs)
        if schema is None:
            assert butler is not None, "Neither butler nor schema is defined"
            schema = butler.get(self.config.coaddName + "Coadd_mergeDet_schema", immediate=True).schema
        self.schemaMapper = afwTable.SchemaMapper(schema)
        self.schemaMapper.addMinimalSchema(schema)
        self.schema = self.schemaMapper.getOutputSchema()
        self.algMetadata = PropertyList()
        if self.config.doDeblend:
            if peakSchema is None:
                assert butler is not None, "Neither butler nor peakSchema is defined"
                peakSchema = butler.get(self.config.coaddName + "Coadd_peak_schema", immediate=True).schema
            self.makeSubtask("deblend", schema=self.schema, peakSchema=peakSchema)
        self.makeSubtask("measurement", schema=self.schema, algMetadata=self.algMetadata)
        self.makeSubtask("setPrimaryFlags", schema=self.schema)
        if self.config.doPropagateFlags:
            self.makeSubtask("propagateFlags", schema=self.schema)
        if self.config.doMatchSources:
            self.makeSubtask("astrometry", schema=self.schema)

    def run(self, patchRef):
        """Measure and deblend"""
        exposure = patchRef.get(self.config.coaddName + "Coadd_calexp", immediate=True)
        sources = self.readSources(patchRef)
        if self.config.doDeblend:
            self.deblend.run(exposure, sources, exposure.getPsf())

            bigKey = sources.schema["deblend.parent-too-big"].asKey()
            numBig = sum((s.get(bigKey) for s in sources)) # catalog is non-contiguous so can't extract column
            if numBig > 0:
                self.log.warn("Patch %s contains %d large footprints that were not deblended" %
                              (patchRef.dataId, numBig))
        self.measurement.run(exposure, sources)
        skyInfo = getSkyInfo(coaddName=self.config.coaddName, patchRef=patchRef)
        self.setPrimaryFlags.run(sources, skyInfo.skyMap, skyInfo.tractInfo, skyInfo.patchInfo,
                                 includeDeblend=self.config.doDeblend)
        if self.config.doPropagateFlags:
            self.propagateFlags.run(patchRef.getButler(), sources, self.propagateFlags.getCcdInputs(exposure),
                                    exposure.getWcs())
        if self.config.doMatchSources:
            self.writeMatches(patchRef, exposure, sources)
        self.write(patchRef, sources)

    def readSources(self, dataRef):
        """Read input sources

        We also need to add columns to hold the measurements we're about to make
        so we can measure in-place.
        """
        merged = dataRef.get(self.config.coaddName + "Coadd_mergeDet", immediate=True)
        self.log.info("Read %d detections: %s" % (len(merged), dataRef.dataId))
        idFactory = self.makeIdFactory(dataRef)
        for s in merged:
            idFactory.notify(s.getId())
        table = afwTable.SourceTable.make(self.schema, idFactory)
        sources = afwTable.SourceCatalog(table)
        sources.extend(merged, self.schemaMapper)
        return sources

    def writeMatches(self, dataRef, exposure, sources):
        """Write matches of the sources to the astrometric reference catalog

        We use the Wcs in the exposure to match sources.

        dataRef: data reference
        exposure: exposure with Wcs
        sources: source catalog
        """
        result = self.astrometry.astrometer.useKnownWcs(sources, exposure=exposure)
        if result.matches:
            matches = afwTable.packMatches(result.matches)
            matches.table.setMetadata(result.matchMetadata)
            dataRef.put(matches, self.config.coaddName + "Coadd_measMatch")
        return result

    def write(self, dataRef, sources):
        """Write the source catalog"""
        dataRef.put(sources, self.config.coaddName + "Coadd_meas")
        self.log.info("Wrote %d sources: %s" % (len(sources), dataRef.dataId))


##############################################################################################################

class MergeMeasurementsConfig(MergeSourcesConfig):
    pseudoFilterList = ListField(dtype=str, default=['sky'],
                                 doc="Names of filters which may have no associated detection\n"
                                     "(N.b. should include MergeDetectionsConfig.skyFilterName)")
    snName = Field(dtype=str, default="flux.psf",
                       doc="Name of flux measurement for calculating the S/N when choosing the "
                           "reference band.")
    minSN = Field(dtype=float, default=10.,
                  doc="If the S/N from the priority band is below this value (and the S/N "
                      "is larger than minSNDiff compared to the priority band), use the band with "
                      "the largest S/N as the reference band."
                  )
    minSNDiff = Field(dtype=float, default=3.,
                  doc="If the difference in S/N between another band and the priority band is larger "
                      "than this value (and the S/N in the priority band is less than minSN) "
                      "use the band with the largest S/N as the reference band"
                  )

class MergeMeasurementsTask(MergeSourcesTask):
    """Measure measurements from multiple bands"""
    _DefaultName = "mergeCoaddMeasurements"
    ConfigClass = MergeMeasurementsConfig
    inputDataset = "meas"
    outputDataset = "ref"
    getSchemaCatalogs = _makeGetSchemaCatalogs("ref")

    def __init__(self, butler=None, schema=None, **kwargs):
        """Initialize the task.

        Additional keyword arguments (forwarded to MergeSourcesTask.__init__):
         - schema: the schema of the detection catalogs used as input to this one
         - butler: a butler used to read the input schema from disk, if schema is None

        The task will set its own self.schema attribute to the schema of the output merged catalog.
        """
        MergeSourcesTask.__init__(self, butler=butler, schema=schema, **kwargs)
        inputSchema = self.getInputSchema(butler=butler, schema=schema)
        self.schemaMapper = afwTable.SchemaMapper(inputSchema)
        self.schemaMapper.addMinimalSchema(inputSchema, True)
        self.fluxKey = inputSchema.find(self.config.snName).key
        self.fluxErrKey = inputSchema.find(self.config.snName + ".err").key
        self.flagKeys = {}
        for band in self.config.priorityList:
            short = getShortFilterName(band)
            outputKey = self.schemaMapper.editOutputSchema().addField(
                "merge.measurement.%s" % short,
                type="Flag",
                doc="Flag field set if the measurements here are from the %s filter" % band
            )
            peakKey = inputSchema.find("merge.peak.%s" % short).key
            footprintKey = inputSchema.find("merge.footprint.%s" % short).key
            self.flagKeys[band] = Struct(peak=peakKey, footprint=footprintKey, output=outputKey)
        self.schema = self.schemaMapper.getOutputSchema()

        self.pseudoFilterKeys = []
        for filt in self.config.pseudoFilterList:
            try:
                self.pseudoFilterKeys.append(self.schema.find("merge.peak.%s" % filt).getKey())
            except Exception as e:
                self.log.warn("merge.peak is not set for pseudo-filter %s" % filt)

    def mergeCatalogs(self, catalogs, patchRef):
        """Merge measurement catalogs to create a single reference catalog for forced photometry

        For parent sources, we choose the first band in config.priorityList for which the
        merge.footprint flag for that band is is True.

        For child sources, the logic is the same, except that we use the merge.peak flags.
        """
        # Put catalogs, filters in priority order
        orderedCatalogs = [catalogs[band] for band in self.config.priorityList if band in catalogs.keys()]
        orderedKeys = [self.flagKeys[band] for band in self.config.priorityList if band in catalogs.keys()]

        mergedCatalog = afwTable.SourceCatalog(self.schema)
        mergedCatalog.reserve(len(orderedCatalogs[0]))

        idKey = orderedCatalogs[0].table.getIdKey()
        for catalog in orderedCatalogs[1:]:
            if numpy.any(orderedCatalogs[0].get(idKey) != catalog.get(idKey)):
                raise ValueError("Error in inputs to MergeCoaddMeasurements: source IDs do not match")

        # This first zip iterates over all the catalogs simultaneously, yielding a sequence of one
        # record for each band, in order.
        for n, orderedRecords in enumerate(zip(*orderedCatalogs)):
            # Now we iterate over those record-band pairs, keeping track of the priority and largest S/N band
            maxSNRecord = None
            maxSNFlagKeys = None
            maxSN = -1.
            priorityRecord = None
            priorityFlagKeys = None
            prioritySN = -1.
            hasPseudoFilter = False

            for inputRecord, flagKeys in zip(orderedRecords, orderedKeys):
                parent = (inputRecord.getParent() == 0 and inputRecord.get(flagKeys.footprint))
                child = (inputRecord.getParent() != 0 and inputRecord.get(flagKeys.peak))

                if not (parent or child):
                    for pseudoFilterKey in self.pseudoFilterKeys:
                        if inputRecord.get(pseudoFilterKey):
                            hasPseudoFilter = True
                            priorityRecord = inputRecord
                            priorityFlagKeys = flagKeys
                            break
                    if hasPseudoFilter:
                        break

                sn = inputRecord.get(self.fluxKey)/inputRecord.get(self.fluxErrKey)
                if numpy.isnan(sn) or sn < 0.:
                    sn = 0
                if (parent or child) and priorityRecord is None:
                    priorityRecord = inputRecord
                    priorityFlagKeys = flagKeys
                    prioritySN = sn
                if sn > maxSN:
                    maxSNRecord = inputRecord
                    maxSNFlagKeys = flagKeys
                    maxSN = sn

            # If the priority band has S/N below the minimum S/N threshold and the largest S/N is larger
            # than the minimum S/N difference threshold, use that one as the reference band.  For pseudo
            # filter objects we choose the first band in the priority list.
            bestRecord = None
            bestFlagKeys = None
            if hasPseudoFilter:
                bestRecord = priorityRecord
                bestFlagKeys = priorityFlagKeys
            if (prioritySN < self.config.minSN and (maxSN - prioritySN) > self.config.minSNDiff and
                maxSNRecord is not None):
                bestRecord = maxSNRecord
                bestFlagKeys = maxSNFlagKeys
            elif priorityRecord is not None:
                bestRecord = priorityRecord
                bestFlagKeys = priorityFlagKeys

            if bestRecord is not None and bestFlagKeys is not None:
                outputRecord = mergedCatalog.addNew()
                outputRecord.assign(bestRecord, self.schemaMapper)
                outputRecord.set(bestFlagKeys.output, True)
            else: # if we didn't find any records
                raise ValueError("Error in inputs to MergeCoaddMeasurements: no valid reference for %s" %
                                 inputRecord.getId())

        copySlots(orderedCatalogs[0], mergedCatalog)

        # more checking for sane inputs, since zip silently iterates over the smallest sequence
        for inputCatalog in orderedCatalogs:
            if len(mergedCatalog) != len(inputCatalog):
                raise ValueError("Mismatch between catalog sizes: %s != %s" %
                                 (len(mergedCatalog), len(orderedCatalogs)))

        return mergedCatalog

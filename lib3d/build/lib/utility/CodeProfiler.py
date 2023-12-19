import cProfile, pstats, io
#from pstats import SortKey

CodeProfiler = None
def StartCodeProfiler():
    global CodeProfiler
    if CodeProfiler != None:
        print("Code profiler has already started!")
    else:
        CodeProfiler = cProfile.Profile()
        CodeProfiler.enable()

def DisplayCodeProfilerResultsAndStopCodeProfiler(MessageToDisplayAboveProfilerResults="\t\t\t\t+++++ CodeProfiler ++++++", top = 25):
    global CodeProfiler
    if CodeProfiler == None:
        print ("Code profiler has not been started yet.")
    else:
        CodeProfiler.disable()
        print(MessageToDisplayAboveProfilerResults)
        stats = pstats.Stats(CodeProfiler).strip_dirs().sort_stats("cumtime")
        stats.print_stats(top)

        del CodeProfiler
        CodeProfiler = None

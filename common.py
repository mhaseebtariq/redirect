import psutil


def get_processes(ids):
    processes = []
    for process in psutil.process_iter():
        cmdline = []
        try:
            cmdline = process.cmdline()
        except:  # noqa
            pass
        if ids.intersection(cmdline):
            processes.append(process)
    return processes

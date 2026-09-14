import multiprocessing


def write_stl_with_timeout(brep, file_path: str, timeout: int):
    """
    Executes the write_stl_files_func in a separate process with a timeout.

    :param write_stl_files_func: The function to execute.
    :param args: Tuple of arguments to pass to the function.
    :param timeout: Timeout in seconds.
    :return: Result of the function, or None if it times out.
    """

    from OCC.Extend.DataExchange import write_stl_file

    def target(result_queue, brep, file_path):
        try:
            result = write_stl_file(
                brep,
                file_path,
                linear_deflection=0.001,
                angular_deflection=0.5,
            )
            # result = write_stl_files_func(*args)
            result_queue.put(result)
        except Exception as e:
            result_queue.put(e)

    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=target, args=(result_queue, brep, file_path)
    )
    process.start()

    process.join(timeout)
    if process.is_alive():
        print(f"Function timed out. Terminating process (PID: {process.pid})...")
        process.terminate()
        process.join()
        return None

    if not result_queue.empty():
        result = result_queue.get()
        if isinstance(result, Exception):
            raise result
        return result
    return None

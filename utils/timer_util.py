import time

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.task_name = ""

    def start(self, task_name=""):
        self.start_time = time.time()
        self.task_name = task_name

    def stop(self):
        self.end_time = time.time()

    def get_elapsed_time(self):
        if self.start_time is None or self.end_time is None:
            raise Exception("Timer has not been started or stopped")
        return self.end_time - self.start_time

    def show_elapsed_time(self):
        elapsed_time = self.get_elapsed_time()
        if self.task_name:
            print(f"{self.task_name} processing time: {elapsed_time:.2f} seconds")
        else:
            print(f"Total processing time: {elapsed_time:.2f} seconds")

    

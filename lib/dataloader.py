import queue

class DataLoader:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.train_queue = queue.Queue()
        self.test_queues = {}

        # Initialize test queues for each label
        for label in self.test_data.keys():
            self.test_queues[label] = queue.Queue()

        # Populate the train queue
        self._populate_train_queue()

    def _populate_train_queue(self):
        for data in self.train_data:
            self.train_queue.put(data)

    def get_train_batch(self, batch_size):
        batch = []
        for _ in range(batch_size):
            try:
                data = self.train_queue.get(timeout=1)
                batch.append(data)
            except queue.Empty:
                break
        return batch

    def get_test_batch(self, label, batch_size):
        with self.lock:
            test_queue = self.test_queues[label]
        batch = []
        for _ in range(batch_size):
            try:
                data = test_queue.get(timeout=1)
                batch.append(data)
            except queue.Empty:
                break
        return batch

    def add_test_data(self, label, data):
        with self.lock:
            test_queue = self.test_queues[label]
        test_queue.put(data)

    def evaluate_test_data(self):
        for label, test_queue in self.test_queues.items():
            while not test_queue.empty():
                data = test_queue.get()
                # Perform evaluation on the test data
                # ...

    def stop(self):
        self.train_thread.join()
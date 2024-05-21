class Cluster:
    def __init__(self, size):
        self.batch_list = [0] * size
        self.bool_list = [False] * size

    def get(self, index):
        return self.zero_list[index], self.bool_list[index]

    def set(self, index, zero_value, bool_value):
        self.zero_list[index] = zero_value
        self.bool_list[index] = bool_value

    def __len__(self):
        return len(True in self.bool_list)
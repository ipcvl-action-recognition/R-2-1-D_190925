class TrainSetting:
    def __init__(self,folder_path, readfile, framefile, clip_len, crop_size):
        self.folder_path = folder_path
        self.readfile = readfile
        self.framefile = framefile
        self.clip_len = clip_len
        self.crop_size = crop_size

    def get_path(self):
        return self.folder_path

    def get_readfile(self):
        return self.readfile

    def get_framefile(self):
        return self.framefile

    def get_clip_len(self):
        return self.clip_len

    def get_crop_size(self):
        return self.crop_size

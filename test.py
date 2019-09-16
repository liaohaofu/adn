from adn.utils import Logger
from adn.tester import Tester
from adn.models import ADNTest
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr


class ADNTester(Tester):
    def __init__(self, **params):
        super(ADNTester, self).__init__(**params)
    
    def get_image(self, data):
        data_opts = self.opts.dataset
        dataset_type = data_opts["dataset_type"]
        if dataset_type == "deep_lesion":
            if data_opts[dataset_type]["load_mask"]:
                return data['lq_image'], data['hq_image'], data["data_name"], data["mask"]
            else:
                return data['lq_image'], data['hq_image'], data["data_name"]
        elif dataset_type == "spineweb":
            return data['a'], data['b'], data["data_name"]
        elif dataset_type == "nature_image":
            return data['artifact'], data['no_artifact'], data["data_name"]

    def get_metric(self, metric):
        def measure(x, y):
            x = self.dataset.to_numpy(x, False)
            y = self.dataset.to_numpy(y, False)
            x = x * 0.5 + 0.5
            y = y * 0.5 + 0.5

            return metric(x, y, data_range=1.0)
        return measure

    def get_pairs(self):
        if hasattr(self.model, 'mask'):
            mask = self.model.mask
            img_low = self.model.img_low * mask
            img_high = self.model.img_high * mask
            pred_lh = self.model.pred_lh * mask
        else:
            img_low = self.model.img_low
            img_high = self.model.img_high
            pred_lh = self.model.pred_lh

        return [
            ("before", (img_low, img_high)), 
            ("after", (pred_lh, img_high))], self.model.name

    def get_visuals(self, n=8):
        lookup = [
            ("l", "img_low"), ("ll", "pred_ll"), ("lh", "pred_lh"),
            ("h", "img_high"), ("hl", "pred_hl"), ("hh", "pred_hh")]
        visual_window = self.opts.visual_window
       
        def visual_func(x):
            x = x * 0.5 + 0.5
            x[x < visual_window[0]] = visual_window[0]
            x[x > visual_window[1]] = visual_window[1]
            x = (x - visual_window[0]) / (visual_window[1] - visual_window[0])
            return x

        return self.model._get_visuals(lookup, n, visual_func, False)

    def get_logger(self, opts):
        self.logger = Logger(self.run_dir, self.epoch, self.run_name)
        self.logger.add_iter_visual_log(self.get_visuals, 1, "test_visuals")
        self.logger.add_metric_log(self.get_pairs,
            (("ssim", self.get_metric(ssim)), ("psnr", self.get_metric(psnr))), opts.metrics_step)

        return self.logger

    def evaluate(self, model, data):
        model.evaluate(*data[:3])
        if len(data) == 4:
            mask = 1 - data[3].to(model.img_high)
            model.mask = mask

if __name__ == "__main__":
    tester = ADNTester(
        name="adn", model_class=ADNTest,
        description="Test an artifact disentanglement network")
    tester.run()

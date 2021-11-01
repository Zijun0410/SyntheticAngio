import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from torchvision.utils import save_image
import os

from pathlib import Path 
import pandas as pd

class EasyTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # only run for one epoch
            # TODO: define the len object in data_loader
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (chek_idx, data, target, _) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            # data.shape =  (16, 1, 512, 512)
            # target = target.unsqueeze(1) # [16, 1, 512, 512]
            # assert len(np.unique(np.array(target.cpu())))==2
            self.optimizer.zero_grad()
            output = self.model(data)
            loss_fn = self.criterion()
            loss = loss_fn(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            # assert len(np.unique(np.array(target.cpu())))==2
            # Image_wise metric update
            for data_idx, _ in enumerate(target):
                output_per_image = output[data_idx].squeeze()
                target_per_image = target[data_idx].squeeze()

                # if len(np.unique(target_per_image.cpu().numpy()))!=2:
                #     print(batch_idx,chek_idx[data_idx])
                #     print(len(np.unique(target_per_image.cpu().numpy())))
                #     continue 
                for met in self.metric_ftns:
                    self.train_metrics.update(met.__name__, met(output_per_image, target_per_image))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,self._progress(batch_idx),loss.item()))
                if not self.config.debug:
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                    self.writer.add_image('target', make_grid(target.cpu(), nrow = 8, normalize = True))
                    self.writer.add_image('output', make_grid(output.cpu(), nrow = 8, normalize = True))

            if batch_idx == self.len_epoch:
                break

        train_result_log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            train_result_log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return train_result_log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (_, data, target, _) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss_fn = self.criterion()
                loss = loss_fn(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

                # Image_wise metric update
                for data_idx, _ in enumerate(data):
                    output_per_image = output[data_idx].squeeze()
                    target_per_image = target[data_idx].squeeze()

                    for met in self.metric_ftns:
                        self.valid_metrics.update(met.__name__, met(output_per_image, target_per_image))
                        
                if not self.config.debug:
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                    self.writer.add_image('target', make_grid(target.cpu(), nrow = 8, normalize = True))
                    self.writer.add_image('output', make_grid(output.cpu(), nrow = 8, normalize = True))
        
        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'sample_number'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.sample_number
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def test(self, module_arch, data_loader, checkpoint_path=None):
        """
        :param checkpoint_path: something like saved/models/DenseNet121_UNet/0819_000755_Angio_stratified_split_DL/
        """
        test_data_loader = data_loader.testing_loader()

        test_metrics = MetricTracker(*[m.__name__ for m in self.metric_ftns], writer=self.writer)

        test_logger = self.config.get_logger('test', self.config['trainer']['verbosity'])
        test_logger.info("Training is down, run for testing...")
        if checkpoint_path is None:
            best_model = self.config.save_dir / 'model_best.pth'
        else:
            best_model = Path(checkpoint_path) / 'model_best.pth'            

        test_logger.info(f'Loading checkpoint of the best model: {best_model} ...')
        test_model = self.config.init_obj('arch', module_arch)
        checkpoint = torch.load(str(best_model))
        state_dict = checkpoint['state_dict']
        if self.config['n_gpu'] > 1:
            test_model = torch.nn.DataParallel(test_model)
        test_model.load_state_dict(state_dict) 

        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_model = test_model.to(device)
        test_model.eval()

        test_metrics.reset()
        run_name = self.config['name']
        output_save_dir = data_loader.get_save_dir("Seg_Result") / f'{run_name}'

        if not os.path.isdir(output_save_dir):
            output_save_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for i, (idx, data, target, image_name) in enumerate(test_data_loader):
                data, target = data.to(device), target.to(device)
                output = test_model(data)

                for data_idx, _ in enumerate(target):
                    output_per_image = output[data_idx].squeeze()
                    target_per_image = target[data_idx].squeeze()
                    raw_per_image = data[data_idx].squeeze()

                    for met in self.metric_ftns:
                        test_metrics.update(met.__name__, met(output_per_image, target_per_image))

                    output_image_save_path = output_save_dir / f'{image_name[data_idx]}_predseg.png'
                    target_image_save_path = output_save_dir / f'{image_name[data_idx]}_trueseg.png'
                    raw_image_save_path = output_save_dir / f'{image_name[data_idx]}_synth.png'

                    save_image(output_per_image, output_image_save_path)
                    save_image(target_per_image, target_image_save_path)
                    save_image(raw_per_image, raw_image_save_path)

        test_result_log = test_metrics.result()
        test_logger.info(test_result_log)

        test_df = pd.DataFrame([test_metrics.result(), test_metrics.std()])

        df_save_path = self.config.save_dir / f'test.csv'
        test_df.to_csv(df_save_path)
        test_logger.info(f'Results saved at {df_save_path}')



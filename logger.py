import logging
import os
import time

class Logger():
    def __init__(self, logdir, run_name):
        self.log_name = logdir + '/' + run_name
        self.tf_writer = None
        self.start_time = time.time()
        self.n_eps = 0

        if not os.path.exists(self.log_name):
            os.makedirs(self.log_name)

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_name + '/logger.log'),
                ],
            datefmt='%Y/%m/%d %I:%M:%S %p'
            )

    def log_episode(self, steps, reward, option_lenghts, ep_steps):
        self.n_eps += 1
        logging.info(f"> ep {self.n_eps} done. total_steps={steps} | reward={reward} | episode_steps={ep_steps} "\
            f"| hours={(time.time()-self.start_time) / 60 / 60:.3f}")
        # TODO: log to tensorflow

    def log_data(self, data):
        # log per step
        # - reward 
        # - entropy
        # - Q values..?
        pass

if __name__=="__main__":
    logger = Logger(logdir='runs/', run_name='test_model-test_env')
    steps = 200 ; reward = 5 ; option_lengths = {opt: np.random.randint(0,5,size=(5)) for opt in range(5)} ; ep_steps = 50
    logger.log_episode(steps, reward, option_lengths, ep_steps)
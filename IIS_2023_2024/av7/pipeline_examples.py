import matplotlib.pyplot as plt
from pykeen.models import TransE
from pykeen.pipeline import pipeline
from pykeen.datasets import FB15k237, Nations
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator
from pykeen.sampling import BasicNegativeSampler

if __name__ == '__main__':
    res = pipeline(dataset='Nations',
                   model='TransE',
                   evaluator='RankBasedEvaluator')

    res.save_to_directory('../results/Nations_TransE')

    res.plot_losses()
    plt.show()

    res2 = pipeline(dataset=Nations,
                    model=TransE,
                    training_loop=SLCWATrainingLoop,
                    negative_sampler=BasicNegativeSampler,
                    evaluator=RankBasedEvaluator)

    res2_df = res2.metric_results.to_df()

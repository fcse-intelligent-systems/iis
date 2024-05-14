from torch.optim import Adam
from pykeen import predict
from pykeen.models import TransE
from pykeen.datasets import FB15k237, Nations
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator

if __name__ == '__main__':
    dataset = Nations()
    print(dataset.training.entity_id_to_label)
    print(dataset.training.relation_id_to_label)
    print(dataset.training.mapped_triples)

    sample_triple = dataset.training.mapped_triples[0].detach().cpu().numpy()
    triple = [dataset.training.entity_id_to_label[sample_triple[0]],
              dataset.training.relation_id_to_label[sample_triple[1]],
              dataset.training.entity_id_to_label[sample_triple[2]]]
    print(sample_triple)
    print(triple)

    model = TransE(triples_factory=dataset.training)
    optimizer = Adam(params=model.get_grad_params())
    trainer = SLCWATrainingLoop(model=model,
                                triples_factory=dataset.training,
                                optimizer=optimizer)
    trainer.train(triples_factory=dataset.training,
                  num_epochs=2,
                  batch_size=64)

    evaluator = RankBasedEvaluator()
    res = evaluator.evaluate(model=model,
                             mapped_triples=dataset.testing.mapped_triples,
                             batch_size=128,
                             additional_filter_triples=[dataset.training.mapped_triples,
                                                        dataset.validation.mapped_triples])
    # Who do we predict brazil participates in inter-governmental organizations with?
    preds = predict.predict_target(model=model,
                                   head="brazil",
                                   relation="intergovorgs",
                                   triples_factory=dataset.training).filter_triples(dataset.testing).df

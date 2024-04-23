import networkx as nx
import ndlib.models.epidemics as ep
import ndlib.models.opinions as op
import ndlib.models.ModelConfig as mc
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
from ndlib.viz.mpl.DiffusionPrevalence import DiffusionPrevalence

if __name__ == '__main__':
    g = nx.erdos_renyi_graph(1000, 0.1)
    nx.to_dict_of_dicts(g)

    model = ep.SIRModel(g)
    print(model.parameters)
    print(model.available_statuses)

    config = mc.Configuration()
    config.add_model_parameter('beta', 0.001)
    config.add_model_parameter('gamma', 0.01)
    config.add_model_parameter("fraction_infected", 0.05)
    model.set_initial_status(config)

    iterations = model.iteration_bunch(200, node_status=True)
    trends = model.build_trends(iterations)

    # compare the trends
    viz = DiffusionTrend(model, trends)
    viz.plot()

    # compare the delta trends
    viz2 = DiffusionPrevalence(model, trends)
    viz2.plot()

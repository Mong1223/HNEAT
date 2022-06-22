"""
Вся логика связанная с самим Hyper NEAT.
"""

import neat


def create_phenotype_network(cppn, substrate, activation_function="sigmoid"):
    """
    Создание рекуррентной сети фенотипа с при помощи CPPN.
    """
    input_coordinates = substrate.input_coordinates
    output_coordinates = substrate.output_coordinates
    # Список координат нейронов в скрытом слое нейронной сети.
    hidden_coordinates = substrate.hidden_coordinates

    input_nodes = list(range(len(input_coordinates)))
    output_nodes = list(range(len(input_nodes), len(
        input_nodes)+len(output_coordinates)))

    counter = 0
    for layer in hidden_coordinates:
        counter += len(layer)

    hidden_nodes = range(len(input_nodes)+len(output_nodes),
                         len(input_nodes)+len(output_nodes)+counter)

    node_evals = []

    # Получение функции активации.
    activation_functions = neat.activations.ActivationFunctionSet()
    activation = activation_functions.get(activation_function)

    # Связывание скрытых нейронов с выходными нейронами.
    counter = 0
    for oc in output_coordinates:
        idx = 0
        for layer in hidden_coordinates:
            im = find_neurons(cppn, oc, layer, hidden_nodes[idx], False)
            idx += len(layer)
            if im:
                #Добавление в список с узлами вычисления, кортежа из номера узла, активации, метода аггрегации, и списка связанных узлов
                node_evals.append(
                    (output_nodes[counter], activation, sum, 0.0, 1.0, im)) 

        counter += 1

    # Связывание скрытых слоёв друг с другом, начиная с того что находится последним к выходному.
    current_layer = 1
    idx = 0
    for layer in hidden_coordinates:
        idx += len(layer)
        counter = idx - len(layer)
        for i in range(current_layer, len(hidden_coordinates)):
            for hc in layer:
                im = find_neurons(
                    cppn, hc, hidden_coordinates[i], hidden_nodes[idx], False)
                if im:
                    node_evals.append(
                        (hidden_nodes[counter], activation, sum, 0.0, 1.0, im))
                counter += 1

            counter -= idx

        current_layer += 1

    # Связывание входного слоя со скрытым.
    counter = 0
    for layer in hidden_coordinates:
        for hc in layer:
            im = find_neurons(cppn, hc, input_coordinates,
                              input_nodes[0], False)
            if im:
                node_evals.append(
                    (hidden_nodes[counter], activation, sum, 0.0, 1.0, im))
            counter += 1

    return neat.nn.RecurrentNetwork(input_nodes, output_nodes, node_evals)


def find_neurons(cppn, coord, nodes, start_idx, outgoing, max_weight=5.0):
    """
    Нахождение веса нейронов с которыми связанна координата.
    """
    im = [] # Список нейронов
    idx = start_idx # Счётчик для нейронов

    for node in nodes:
        # Нахождение веса связи между нейронами в фенотипе
        w = query_cppn(coord, node, outgoing, cppn, max_weight)

        if w != 0.0:  # Включает нейрон в список только если связь не равна нулю.
            im.append((idx, w))
        idx += 1

    return im


def query_cppn(coord1, coord2, outgoing, cppn, max_weight=5.0):
    """
    Функция нахождения веса между связями в фенотипе.
    Вес возвращается на решении о том, это прямая или обратная связб.
    """

    if outgoing:
        i = [coord1[0], coord1[1], coord2[0], coord2[1], 1.0] # Связь обратная 
    else:
        i = [coord2[0], coord2[1], coord1[0], coord1[1], 1.0] # Связь прямая
    w = cppn.activate(i)[0]
    if abs(w) > 0.2:  # Если связь не прошла порог, считаем что нейроны не связанны.
        if w > 0:
            w = (w - 0.2) / 0.8
        else:
            w = (w + 0.2) / 0.8
        return w * max_weight
    else:
        return 0.0

"""
Главный файл.
"""

import pickle
from pyexpat.errors import XML_ERROR_INCOMPLETE_PE
#Используется библиотечный neat
import neat
import neat.nn
#Вообще это для визуализации, но чёт она не встала
from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate
from pureples.hyperneat.hyperneat import create_phenotype_network

# Network inputs and expected outputs.

NUM_INPUTS = 9
NUM_OUTPUTS = 2

training_examples=350
validation_examples=175
test_examples=174

INPUTS = []
OUTPUTS = []

INPUT_COORDINATES = [( round(-NUM_INPUTS / 2) + i, -1) for i in range(NUM_INPUTS + 1)]
HIDDEN_COORDINATES = [[(round(-NUM_INPUTS / 2) + i, 0) for i in range(NUM_INPUTS + 1)]]
OUTPUT_COORDINATES = [(-1.0, 1.0), (0.0, 1.0)]
ACTIVATIONS = len(HIDDEN_COORDINATES) + 2

SUBSTRATE = Substrate(
    INPUT_COORDINATES, OUTPUT_COORDINATES, HIDDEN_COORDINATES)

# Конфигурация сети CPPN, являющейся субстратом всей hyperNEAT.
CONFIG = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_cppn_experiment')
#Парсер для файла с данными
def parse(FILENAME):
        FILE = open(FILENAME, 'r')
        inputs = []
        outputs = []
        training_examples=182
        validation_examples=91
        for iter, line in enumerate(FILE):
            if iter > 7:
                vars = line.split(' ')
                input_list = [float(vars[i]) for i in range(NUM_INPUTS)]
                output_list = [int(vars[j + NUM_INPUTS]) for j in range(NUM_OUTPUTS)]
                inputs.append(tuple(input_list))
                outputs.append(tuple(output_list))
        test_input = list(inputs[training_examples + validation_examples:])
        test_output = list(outputs[training_examples + validation_examples:])
        
        return test_input, test_output

def eval_fitness(genomes, config):
    """
    Функция приспособленности.
    Для каждого генома вычисляется своя приспособленность, вычисляется как среднеквадратичная ошибка.
    """
    for _, genome in genomes:
        cppn = neat.nn.FeedForwardNetwork.create(genome, config) #Создаётся сеть CPPN для каждого генома NEAT в генотипе
        net = create_phenotype_network(cppn, SUBSTRATE) #Создаётся фенотип на основе CPPN

        sum_square_error = 0.0 #Ошибка

        for inputs, expected in zip(INPUTS, OUTPUTS): #Проход по данным
            new_input = inputs + (1.0,) #Добавляется смещение
            net.reset()

            for _ in range(ACTIVATIONS):#Сеть проходит по всем функциям активации формируя выход
                output = net.activate(new_input)

            #Вычисление ошибки
            sum_square_error += (((output[0] - expected[0])**2 + (output[1] - expected[1])**2) / 2) / len(INPUTS)

        genome.fitness = 1 - sum_square_error


def run(gens):
    """
    Создание популяции и проход сети, подавая функцию eval_fitness как функцию приспособленности.
    Возвращает лучший геном и статистику по нему.
    """
    pop = neat.population.Population(CONFIG) #Создаётся популяция на основе конфига, популяция сетей CPPN
    stats = neat.statistics.StatisticsReporter() #Для сбора статистики
    pop.add_reporter(stats)
    pop.add_reporter(neat.reporting.StdOutReporter(True))

    winner = pop.run(eval_fitness, gens) #Проход популяции CPPN через NEAT, для нахождения лучшего
    print("hyperneat done")
    return winner, stats


# Главная функция
if __name__ == '__main__':
    INPUTS, OUTPUTS = parse("cancer1.dt")
    WINNER = run(300)[0]  # Возвращается самый лучший
    print('\nBest genome:\n{!s}'.format(WINNER))

    # Тестирование сети на основе обучающей выборки.
    print('\nOutput:')
    CPPN = neat.nn.FeedForwardNetwork.create(WINNER, CONFIG)
    WINNER_NET = create_phenotype_network(CPPN, SUBSTRATE)

    for inputs, expected in zip(INPUTS, OUTPUTS):
        new_input = inputs + (1.0,)
        WINNER_NET.reset()

        for i in range(ACTIVATIONS):
            output = WINNER_NET.activate(new_input)

        print("  input {!r}, expected output {!r}, got {!r}".format(
            inputs, expected, output))

    # Сохранение стеи CPPN.
    with open('hyperneat_xor_cppn.pkl', 'wb') as output:
        pickle.dump(CPPN, output, pickle.HIGHEST_PROTOCOL)
    # Неработающая визуализация
#    draw_net(CPPN, filename="hyperneat_xor_cppn")
#    draw_net(WINNER_NET, filename="hyperneat_xor_winner")

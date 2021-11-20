import numpy as np
import matplotlib.pyplot as plt
from math import factorial
import simpy


# СМО
class QueuingSystem:
    def __init__(self,
                 request_flow_rate,
                 service_flow_rate,
                 max_queue_request_length,
                 queue_waiting_param,
                 number_channels,
                 env):
        # lambda (интенсивность постуления заявок == сколько заявок в среднем поступает на обслуживание за единицу времени)
        self.request_flow_rate = request_flow_rate
        # mu (интенсивность потока обслуживания == среднее число заявок, обсулиживаемое за единицу времени)
        self.service_flow_rate = service_flow_rate
        # m (Максимальное число заявок в очереди)
        self.max_queue_request_length = max_queue_request_length
        # v - (Время пребывания заявки в очереди ограничено некоторым случайным сроком T, распределенным по показательному закону с параметром v)
        self.queue_waiting_param = queue_waiting_param

        self.number_channels = number_channels

        # Сколько пробыла в СМО каждая заявка
        self.request_in_qs_time = []
        # Сколько заявок сейчас в СМО в очереди и обрабатывается
        self.queuing_and_processing_request = []
        # Сколько пробыла каждая из заявок в очереди
        self.request_in_qw_time = []
        # Склоько заявок в очереди в текущий момент времени
        self.request_amount_in_query = []
        # Сколько заявок было обработано в текущий момент времени
        self.requests_processed = []
        # Скольким заявкам было отказано в текущий момент времени
        self.requests_rejected = []

        self.env = env
        self.channel = simpy.Resource(env, number_channels)

    def application_processing(self):
        yield self.env.timeout(np.random.exponential(1 / self.service_flow_rate))

    def application_waiting(self):
        yield self.env.timeout(np.random.exponential(1 / self.queue_waiting_param))

    def __run_process(self):
        while True:
            yield self.env.timeout(np.random.exponential(1 / self.request_flow_rate))
            self.env.process(self.__add_request())

    def run_model(self, minutes):
        self.env.process(self.__run_process())
        # Силуляция будет работать 100 минут
        self.env.run(minutes)

    def __add_request(self):
        # Количество заявок, который сейчас обрабатываются (Number of users currently using the resource)
        busy_channels = self.channel.count
        # Текущее количество заявок в очереди
        requests_in_queue = len(self.channel.queue)

        self.request_amount_in_query.append(requests_in_queue)
        self.queuing_and_processing_request.append(requests_in_queue + busy_channels)

        with self.channel.request() as request:
            current_busy_channels = self.channel.count
            current_requests_in_queue = len(self.channel.queue)
            if current_requests_in_queue > self.max_queue_request_length:
                self.requests_rejected.append(self.max_queue_request_length + self.number_channels + 1)
                self.request_in_qs_time.append(0)
                self.request_in_qw_time.append(0)
            else:
                start_time = self.env.now
                self.requests_processed.append(current_busy_channels + current_requests_in_queue)
                res = yield request | self.env.process(self.application_waiting())
                self.request_in_qw_time.append(self.env.now - start_time)
                if request in res:
                    yield self.env.process(self.application_processing())
                self.request_in_qs_time.append(self.env.now - start_time)


def find_mean(text, data):
    mean = np.array(data).mean()
    print(text, mean)
    return mean


def show_empirical_probabilities(queuing_system,
                                 num_channel,
                                 max_queue_request_length,
                                 request_flow_rate,
                                 service_flow_rate):
    print('-------------------------Эмпирические данные------------------------')
    requests_processed = np.array(queuing_system.requests_processed)
    queuing_and_processing_request = np.array(queuing_system.queuing_and_processing_request)
    requests_rejected_len = len(queuing_system.requests_rejected)

    requests_amount = len(requests_processed) + requests_rejected_len
    P = [len(requests_processed[requests_processed == value]) / requests_amount for value in
         range(1, num_channel + max_queue_request_length + 1)]
    for index, p in enumerate(P):
        print(f'P{index} = {p}')
    P_reject = requests_rejected_len / requests_amount
    Q = 1 - P_reject
    A = request_flow_rate * Q
    print('Относительная пропусная способность Q:', Q)
    print('Абсолютная пропусная способность A:', A)
    print('Вероятность отказа:', P_reject)
    average_amount_requests_in_queue = find_mean("Средняя число заявок в очереди:",
                                                 queuing_system.request_amount_in_query)
    average_amount_requests_in_qs = find_mean("Среднее число заявок в СМО:", queuing_and_processing_request)
    average_request_time_in_queue = find_mean("Среднее время пребывания заявки в очереди:",
                                              queuing_system.request_in_qw_time)
    average_request_time_in_qs = find_mean("среднее время пребывания заявки в СМО:", queuing_system.request_in_qs_time)
    average_amount_busy_channels = Q * request_flow_rate / service_flow_rate
    print('Среднее количество занятых каналов:', average_amount_busy_channels)
    return P, [Q, A, average_amount_requests_in_queue, average_amount_requests_in_qs, average_request_time_in_queue,
               average_request_time_in_qs, average_amount_busy_channels]


def show_theoretical_probabilities(request_flow_rate,
                                   service_flow_rate,
                                   num_channel,
                                   queue_waiting_param,
                                   max_queue_request_length):
    def find_average_amount_requests_in_qs():
        kp_sum = sum([k * p0 * (ro ** k) / factorial(k) for k in range(1, num_channel + 1)])
        np_sum = 0
        for i in range(1, max_queue_request_length + 1):
            n_t_b = []
            for t in range(1, i + 1):
                n_t_b.append(num_channel + t * betta)
            np_sum += (num_channel + i) * pn * ro ** i / np.prod(n_t_b)
        return kp_sum + np_sum

    print('-------------------------Теоретические данные------------------------')
    P_theor = []
    betta = queue_waiting_param / service_flow_rate
    ro = request_flow_rate / service_flow_rate  # Коэффициент загрузки СМО
    # вероятность, что канал свободен
    p0 = (sum([ro ** i / factorial(i) for i in range(num_channel + 1)]) +
          (ro ** num_channel / factorial(num_channel)) *
          sum([ro ** i / (np.prod([num_channel + t * betta for t in range(1, i + 1)]))
               for i in range(1, max_queue_request_length + 1)])) ** -1
    P = 0
    P += p0
    for i in range(0, num_channel + 1):
        px = (ro ** i / factorial(i)) * p0
        P += px
        P_theor.append(px)
        print(f'P{i} = {px}')
    pn = px
    for i in range(1, max_queue_request_length):
        px = (ro ** i / np.prod([num_channel + t * betta for t in range(1, i + 1)])) * pn
        P += px
        P_theor.append(px)
        print(f'P{num_channel + i} = {px}')
    P = px
    Q = 1 - P  # вероятность обслуживания поступившей в СМО заявки
    A = Q * request_flow_rate  # среднее число заявок, обслуживаемых в СМО в единицу временени
    average_amount_requests_in_queue = sum(
        [i * pn * (ro ** i) / np.prod([num_channel + t * betta for t in range(1, i + 1)]) for
         i in range(1, max_queue_request_length + 1)])
    average_amount_requests_in_qs = find_average_amount_requests_in_qs()
    average_request_time_in_queue = average_amount_requests_in_queue / request_flow_rate
    average_request_time_in_qs = average_amount_requests_in_qs / request_flow_rate
    average_amount_busy_channels = Q * ro
    print('Относительная пропусная способность Q:', Q)
    print('Абсолютная пропусная способность A:', A)
    print('Вероятность отказа:', P)
    print('Средняя число заявок в очереди:', average_amount_requests_in_queue)
    print('Среднее число заявок в СМО:', average_amount_requests_in_qs)
    print('Среднее время пребывания заявки в очереди:', average_request_time_in_queue)
    print('Среднее время пребывания заявки в СМО: ', average_request_time_in_qs)
    print('Среднее количество занятых каналов:', average_amount_busy_channels)
    return P_theor, [Q, A, average_amount_requests_in_queue, average_amount_requests_in_qs,
                     average_request_time_in_queue,
                     average_request_time_in_qs, average_amount_busy_channels]


def compare_empirical_and_theoretical_statistics(empirical_statistics, theoretical_statistics):
    index = np.arange(len(empirical_statistics))
    bar_width = 0.35
    opacity = 0.8

    plt.bar(index, empirical_statistics, bar_width,
            alpha=opacity,
            color='b',
            label='Эмпирические')

    plt.bar(index + bar_width, theoretical_statistics, bar_width,
            alpha=opacity,
            color='g',
            label='Теоретические')

    plt.title('Сравнение эмпирических и теоретических данных')
    plt.xticks(index + bar_width, (
        "Q", "A", "Avr. queue requests", "Avg QS requests", "Avg queue reqst. time",
        "Avg. reqst. qs time", "Avg. busy channels"), rotation='vertical')
    plt.legend()

    plt.tight_layout()
    plt.show()


def compare_empirical_and_theoretical_final_probabilities(P_emp, P_theor):
    index = np.arange(len(P_emp))
    bar_width = 0.35
    opacity = 0.8

    plt.bar(index, P_emp, bar_width,
            alpha=opacity,
            color='b',
            label='Эмпирические финальные вероятности')

    plt.bar(index + bar_width, P_theor, bar_width,
            alpha=opacity,
            color='g',
            label='Теоретические финальные вероятности')

    plt.title('Сравнение эмпирических и теоретических финальных вероятностей')
    plt.xticks(index + bar_width, [f"P{i}" for i in range(len(P_emp))], rotation='vertical')
    plt.legend()

    plt.tight_layout()
    plt.show()


def show_test_data(num_channels,
                   request_flow_rate,
                   service_flow_rate,
                   queue_waiting_param,
                   max_queue_request_length):
    print('Количество каналов (n):', num_channels)
    print('Интенсивность потока заявок (lambda):', request_flow_rate)
    print('Интенсивность потока обслуживания (mu):', service_flow_rate)
    print('Времени пребывания в очереди (v):', queue_waiting_param)
    print('Размер очереди (m):', max_queue_request_length)

def show_stationare_plots(qs, theoretical_probabilities, interval_count):
    intervals = np.array_split(qs.queuing_and_processing_request, interval_count)
    for i in range(1, len(intervals)):
        intervals[i] = np.append(intervals[i], intervals[i - 1])
    for i in range(len(theoretical_probabilities)):
        interval_probabilities = []
        for interval in intervals:
            interval_probabilities.append(len(interval[interval == i]) / len(interval))
        plt.figure(figsize=(5, 5))
        plt.bar(range(len(interval_probabilities)), interval_probabilities)
        plt.title(f"Probabilitiy {i}")
        plt.axhline(y=theoretical_probabilities[i], xmin=0, xmax=len(interval_probabilities), color='red')
        plt.show()

def run_test(num_channels, service_flow_rate, request_flow_rate, queue_waiting_param, max_queue_request_length,
             running_time):
    show_test_data(num_channels=num_channels,
                   service_flow_rate=service_flow_rate,
                   request_flow_rate=request_flow_rate,
                   queue_waiting_param=queue_waiting_param,
                   max_queue_request_length=max_queue_request_length)

    queuing_system = QueuingSystem(number_channels=num_channels,
                                   service_flow_rate=service_flow_rate,
                                   request_flow_rate=request_flow_rate,
                                   queue_waiting_param=queue_waiting_param,
                                   max_queue_request_length=max_queue_request_length,
                                   env=simpy.Environment())

    queuing_system.run_model(minutes=running_time)

    P_emp, empirical_statistics = show_empirical_probabilities(queuing_system=queuing_system,
                                                               num_channel=num_channels,
                                                               max_queue_request_length=max_queue_request_length,
                                                               request_flow_rate=request_flow_rate,
                                                               service_flow_rate=service_flow_rate)

    P_theor, theoretical_statistics = show_theoretical_probabilities(request_flow_rate=request_flow_rate,
                                                                     service_flow_rate=service_flow_rate,
                                                                     num_channel=num_channels,
                                                                     queue_waiting_param=queue_waiting_param,
                                                                     max_queue_request_length=max_queue_request_length)

    compare_empirical_and_theoretical_final_probabilities(P_emp=P_emp, P_theor=P_theor)
    compare_empirical_and_theoretical_statistics(empirical_statistics=empirical_statistics,
                                                 theoretical_statistics=theoretical_statistics)
    show_stationare_plots(qs=queuing_system, interval_count=100, theoretical_probabilities=P_theor)


if __name__ == '__main__':
    run_test(num_channels=2,
             service_flow_rate=5,
             request_flow_rate=10,
             queue_waiting_param=1,
             max_queue_request_length=10,
             running_time=6000)

    print("--------------------------------------------------------------------------")
    run_test(num_channels=5,
             service_flow_rate=10,
             request_flow_rate=100,
             queue_waiting_param=1,
             max_queue_request_length=10,
             running_time=6000)

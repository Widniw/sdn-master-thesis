class Switch:
    def __init__(self, service_rate, queue_capacity, queue_length = 0, aggr_arrival_rate = 0):
        self.service_rate = service_rate
        self.queue_capacity = queue_capacity
        self.queue_length = queue_length
        self.aggr_arrival_rate = aggr_arrival_rate
        self.traffic_intensity = self.aggr_arrival_rate / self.service_rate
        self.packet_loss_probabilty = ((1 - self.traffic_intensity) * (self.traffic_intensity)**(self.queue_capacity)) \
                                    / (1 - (self.traffic_intensity)**(self.queue_capacity + 1))
        self.exp_queue_occupation = (self.traffic_intensity/(1 - self.traffic_intensity)) - \
                                    ((self.queue_capacity + 1) * self.traffic_intensity**(self.queue_capacity + 1)) \
                                    /(1 - self.traffic_intensity**(self.queue_capacity + 1))
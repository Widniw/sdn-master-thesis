# def traffic_leaving_mm1k(incoming_flows, service_rate, queue_capacity):
#     aggregated_traffic = sum(incoming_flows)

#     ro = aggregated_traffic / service_rate

#     leaving_propability = (1 - ro**queue_capacity) / (1 - ro**(queue_capacity+1))
#     print(f"{leaving_propability = }")

    
#     leaving_flows = [flow * leaving_propability for flow in incoming_flows]

#     return leaving_flows


def traffic_leaving_mm1k(incoming_flows, service_rate, queue_capacity):
    aggregated_traffic = 0
    for edge_flows in incoming_flows.values():
        aggregated_traffic += sum(edge_flows.values())

    # Jeśli nie ma ruchu, zwracamy puste/zerowe flows bez liczenia
    if aggregated_traffic == 0:
        return incoming_flows

    # 2. Obliczenie obciążenia (rho)
    ro = aggregated_traffic / service_rate

    if ro == 1:
        ro = 0.999

    # Unikanie overflow
    if ro < 1.0:
        leaving_probability = (1 - ro**queue_capacity) / (1 - ro**(queue_capacity + 1))
            
    else:
        numerator = (1.0 / ro) - (ro ** -(queue_capacity + 1))
        denominator = 1.0 - (ro ** -(queue_capacity + 1))
        leaving_probability = numerator / denominator    
    
    outgoing_flows = {}
    
    for edge, flows_dict in incoming_flows.items():
        new_flows_dict = {}
        for flow_id, traffic in flows_dict.items():
            new_flows_dict[flow_id] = traffic * leaving_probability
    
        outgoing_flows[edge] = new_flows_dict

    return outgoing_flows


# leaving_flows = traffic_leaving_mm1k([1.34, 3], 4, 2)
# print(f"{leaving_flows = }")



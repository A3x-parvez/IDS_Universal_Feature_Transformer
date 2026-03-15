import os
import csv
import torch
import numpy as np
from datetime import datetime


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clean_column_names(df):
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace("/", "_")
                  .str.replace("-", "_")
    )
    return df

FEATURE_SYNONYMS = {

    # ===============================
    # FLOW LEVEL
    # ===============================
    "dur": "flow_total_duration",
    "flow_duration_milliseconds": "flow_total_duration",
    "flow_duration": "flow_total_duration",

    "rate": "flow_packets_per_second",
    "flow_packets_s": "flow_packets_per_second",
    "flow_packets/s": "flow_packets_per_second",

    "flow_bytes_s": "flow_bytes_per_second",
    "flow_bytes/s": "flow_bytes_per_second",

    # ===============================
    # PORTS
    # ===============================
    "destination_port": "destination_port_number",
    "l4_dst_port": "destination_port_number",
    "l4_src_port": "source_port_number",

    # ===============================
    # PACKET COUNTS
    # ===============================
    "spkts": "forward_total_packet_count",
    "dpkts": "backward_total_packet_count",
    "total_fwd_packets": "forward_total_packet_count",
    "total_backward_packets": "backward_total_packet_count",
    "in_pkts": "forward_total_packet_count",
    "out_pkts": "backward_total_packet_count",

    # ===============================
    # BYTE COUNTS
    # ===============================
    "sbytes": "forward_total_byte_count",
    "dbytes": "backward_total_byte_count",
    "total_length_of_fwd_packets": "forward_total_byte_count",
    "total_length_of_bwd_packets": "backward_total_byte_count",
    "in_bytes": "forward_total_byte_count",
    "out_bytes": "backward_total_byte_count",

    # ===============================
    # PACKET SIZE STATISTICS (FORWARD)
    # ===============================
    "fwd_packet_length_max": "forward_packet_length_maximum",
    "fwd_packet_length_min": "forward_packet_length_minimum",
    "fwd_packet_length_mean": "forward_packet_length_mean",
    "fwd_packet_length_std": "forward_packet_length_standard_deviation",

    # ===============================
    # PACKET SIZE STATISTICS (BACKWARD)
    # ===============================
    "bwd_packet_length_max": "backward_packet_length_maximum",
    "bwd_packet_length_min": "backward_packet_length_minimum",
    "bwd_packet_length_mean": "backward_packet_length_mean",
    "bwd_packet_length_std": "backward_packet_length_standard_deviation",

    # ===============================
    # FLOW PACKET SIZE STATISTICS
    # ===============================
    "max_packet_length": "flow_packet_length_maximum",
    "min_packet_length": "flow_packet_length_minimum",
    "packet_length_mean": "flow_packet_length_mean",
    "packet_length_std": "flow_packet_length_standard_deviation",

    # ===============================
    # INTER ARRIVAL TIME (FORWARD)
    # ===============================
    "sinpkt": "forward_interarrival_time_mean",
    "fwd_iat_mean": "forward_interarrival_time_mean",
    "fwd_iat_std": "forward_interarrival_time_standard_deviation",
    "fwd_iat_min": "forward_interarrival_time_minimum",
    "fwd_iat_max": "forward_interarrival_time_maximum",
    "fwd_iat_total": "forward_interarrival_time_total",

    # ===============================
    # INTER ARRIVAL TIME (BACKWARD)
    # ===============================
    "dinpkt": "backward_interarrival_time_mean",
    "bwd_iat_mean": "backward_interarrival_time_mean",
    "bwd_iat_std": "backward_interarrival_time_standard_deviation",
    "bwd_iat_min": "backward_interarrival_time_minimum",
    "bwd_iat_max": "backward_interarrival_time_maximum",
    "bwd_iat_total": "backward_interarrival_time_total",

    "src_to_dst_iat_min": "forward_interarrival_time_minimum",
    "src_to_dst_iat_max": "forward_interarrival_time_maximum",
    "src_to_dst_iat_avg": "forward_interarrival_time_mean",
    "src_to_dst_iat_stddev": "forward_interarrival_time_standard_deviation",

    "dst_to_src_iat_min": "backward_interarrival_time_minimum",
    "dst_to_src_iat_max": "backward_interarrival_time_maximum",
    "dst_to_src_iat_avg": "backward_interarrival_time_mean",
    "dst_to_src_iat_stddev": "backward_interarrival_time_standard_deviation",

    # ===============================
    # TTL
    # ===============================
    "sttl": "source_time_to_live_value",
    "dttl": "destination_time_to_live_value",
    "min_ttl": "flow_minimum_time_to_live",
    "max_ttl": "flow_maximum_time_to_live",

    # ===============================
    # TCP TIMING
    # ===============================
    "tcprtt": "tcp_round_trip_time",
    "synack": "tcp_syn_ack_delay_time",
    "ackdat": "tcp_ack_data_delay_time",

    "tcp_win_max_in": "tcp_maximum_window_size_forward",
    "tcp_win_max_out": "tcp_maximum_window_size_backward",

    # ===============================
    # THROUGHPUT
    # ===============================
    "src_to_dst_avg_throughput": "forward_average_throughput",
    "dst_to_src_avg_throughput": "backward_average_throughput",

    # ===============================
    # SERVICE / PROTOCOL
    # ===============================
    "protocol": "internet_protocol_identifier",
    "tcp": "is_tcp_protocol",
    "udp": "is_udp_protocol",
    "http": "is_http_application_protocol",

}

def normalize_feature_names(dfs, synonyms):
    for i in range(len(dfs)):
        dfs[i].rename(columns=synonyms, inplace=True)
    print("Feature synonyms applied")
    return dfs


def create_experiment_run(experiment_name):

    base_dir = os.path.join("results", "experiments", experiment_name)
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = os.path.join(base_dir, f"run_{timestamp}")

    graphs_dir = os.path.join(run_dir, "graphs")
    models_dir = os.path.join(run_dir, "models")

    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    return run_dir, graphs_dir, models_dir



def update_experiment_summary(experiment, run_name, history, metrics):

    summary_file = os.path.join("results", "experiment_summary.csv")

    file_exists = os.path.isfile(summary_file)

    with open(summary_file, "a", newline="") as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "experiment",
                "run",

                "final_train_loss",
                "final_train_accuracy",
                "final_train_f1",

                "best_val_loss",
                "best_val_accuracy",
                "best_val_f1",

                "test_accuracy",
                "test_f1",
                "test_precision",
                "test_recall",

                "training_time"
            ])

        writer.writerow([
            experiment,
            run_name,

            history["train_loss"][-1],
            history["train_accuracy"][-1],
            history["train_f1"][-1],

            min(history["val_loss"]),
            max(history["val_accuracy"]),
            max(history["val_f1"]),

            metrics["test_accuracy"],
            metrics["test_f1"],
            metrics["test_precision"],
            metrics["test_recall"],

            metrics["training_time"]
        ])
import re
import matplotlib.pyplot as plt


def plot_losses_with_min_val_line(log_file_path, type="loss"):
    with open(log_file_path, "r") as file:
        log_content = file.readlines()

    if type == "loss":
        target_lines = [line for line in log_content if "Loss" in line]

        train_pattern = re.compile(r"Training Loss:\s*([0-9.]+)")
        val_pattern = re.compile(r"Validation Loss:\s*([0-9.]+)")
        y_label = "Loss"
        title = "Training and Validation Loss"
        save_name = "Loss.jpg"
    elif type == "acc":
        target_lines = [line for line in log_content if "Accuracy" in line]

        train_pattern = re.compile(r"Training Accuracy:\s*([0-9.]+)")
        val_pattern = re.compile(r"Validation Accuracy:\s*([0-9.]+)")
        y_label = "Accuracy"
        title = "Training and Validation Accuracy"
        save_name = "Accuracy.jpg"
    else:
        raise ValueError("Invalid type. Please choose 'loss' or 'acc'.")

    train_values = []
    val_values = []

    for line in target_lines:
        train_match = train_pattern.search(line)
        val_match = val_pattern.search(line)

        if train_match and val_match:
            train_values.append(float(train_match.group(1)))
            val_values.append(float(val_match.group(1)))

    if type == "loss":
        val_value = min(val_values)
        val_epoch = val_values.index(val_value) + 1
    elif type == "acc":
        val_value = max(val_values)
        val_epoch = val_values.index(val_value) + 1

    plt.figure(figsize=(10, 6))
    plt.plot(train_values, label=f"Training {y_label}")
    plt.plot(val_values, label=f"Validation {y_label}")
    plt.axvline(x=val_epoch - 1, color="red", linestyle="--", label=f"Lowest Val {y_label} (Epoch {val_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_name)
    # plt.show()

    return val_value, val_epoch


log_file_path = r"C:\workspace\FASHION-HOW\subtask2\check_points\Best\20240812_175729_862\TASK2_main.log"
plot_losses_with_min_val_line(log_file_path, "loss")
plot_losses_with_min_val_line(log_file_path, "acc")

import csv
import os

from matplotlib import pyplot as plt


def save_to_csv(data, file_path):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss', 'Accuracy'])  # Writing header
        writer.writerows(data)


def load_from_csv(file_path):
    data = []
    if os.path.exists(file_path):
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            for row in reader:
                epoch, loss, accuracy = int(row[0]), float(row[1]), float(row[2])
                data.append((epoch, loss, accuracy))
    return data


def plot_graph(epoch_data, args):
    # Ensure the output directory exists
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract data for plotting
    epochs = [data[0] for data in epoch_data]
    losses = [data[1] for data in epoch_data]
    accuracies = [data[2] for data in epoch_data]
    learning_rates = [data[3] for data in epoch_data]  # Extract learning rates

    # Language mapping based on args
    language_map = {"en": "English", "am": "Amharic"}
    source_language = language_map.get(args.source_lang, args.source_lang)
    target_language = language_map.get(args.target_lang, args.target_lang)

    # Generate plot labels based on source and target languages
    loss_label = f"{source_language} to {target_language} Bert-fused Training Loss"
    accuracy_label = f"{source_language} to {target_language} Bert-fused Training Accuracy"
    lr_label = f"{source_language} to {target_language} Bert-fused Learning Rate Schedule"

    # Plot loss against epochs
    plt.figure(figsize=(6, 5))
    plt.plot(epochs, losses, '-o', label='loss')
    plt.title(f'{loss_label}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_loss.png"))  # Save the loss plot
    plt.close()  # Close the current plot

    # Plot accuracy against epochs
    plt.figure(figsize=(6, 5))
    plt.plot(epochs, accuracies, '-o', label='accuracy')
    plt.title(f'{accuracy_label}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_accuracy.png"))  # Save the accuracy plot
    plt.close()  # Close the current plot

    # Plot learning rate against epochs
    plt.figure(figsize=(6, 5))
    plt.plot(epochs, learning_rates, '-o', label='learning rate', color='green')
    plt.title(f'{lr_label}')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_rate.png"))  # Save the learning rate plot
    plt.close()  # Close the current plot

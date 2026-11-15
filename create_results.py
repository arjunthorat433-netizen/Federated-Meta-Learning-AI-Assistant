import matplotlib.pyplot as plt
import os

clients = ["Client 1", "Client 2", "Client 3"]

federated = [69.00, 66.50, 65.50]
maml = [79.00, 78.50, 78.00]

federated_avg = sum(federated) / len(federated)
maml_avg = sum(maml) / len(maml)

os.makedirs("results", exist_ok=True)

# Save accuracy results
with open("results/accuracy_results.txt", "w") as file:
    file.write("FEDERATED META-LEARNING PROJECT\n")
    file.write("FINAL INDEPENDENT TEST RESULTS\n\n")

    file.write("Federated Learning:\n")
    for client, score in zip(clients, federated):
        file.write(f"{client}: {score:.2f}%\n")
    file.write(f"Average Accuracy: {federated_avg:.2f}%\n\n")

    file.write("MAML:\n")
    for client, score in zip(clients, maml):
        file.write(f"{client}: {score:.2f}%\n")
    file.write(f"Average Accuracy: {maml_avg:.2f}%\n\n")

    file.write(f"MAML Improvement: {maml_avg - federated_avg:.2f} percentage points\n")
    file.write("Graph: accuracy_comparison.png\n")

# Create comparison graph
x = range(len(clients))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width/2 for i in x],
    federated,
    width,
    label="Federated Learning"
)

plt.bar(
    [i + width/2 for i in x],
    maml,
    width,
    label="MAML"
)

plt.xlabel("Clients")
plt.ylabel("Accuracy (%)")
plt.title("Federated Learning vs MAML - Independent Test")
plt.xticks(list(x), clients)
plt.ylim(0, 100)
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.savefig(
    "results/accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Accuracy Results and Graph Updated Successfully!")
print(f"Federated Average: {federated_avg:.2f}%")
print(f"MAML Average: {maml_avg:.2f}%")

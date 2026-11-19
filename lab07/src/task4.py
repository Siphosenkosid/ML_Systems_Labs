
import csv
import time
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F

from task2 import (
    BaselineCNN, DEVICE, MODELS, RESULTS,
    train_loader, test_loader
)

# Settings
SEED = 42
EPOCHS = 5
LR = 0.001
TEMPERATURE = 4
ALPHA = 0.5

torch.manual_seed(SEED)


# Smaller student CNN (~106k parameters)
class StudentCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


def train_student(model, teacher=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            student_logits = model(images)

            ce_loss = F.cross_entropy(
                student_logits, labels
            )

            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(images)

                # Softened probability distributions
                student_probs = F.softmax(
                    student_logits / TEMPERATURE, dim=1
                )

                teacher_log_probs = F.log_softmax(
                    teacher_logits / TEMPERATURE, dim=1
                )

                # KL(student || teacher), as written
                # in the lab PDF
                kd_loss = F.kl_div(
                    teacher_log_probs,
                    student_probs,
                    reduction="batchmean"
                )

                loss = (
                    ALPHA * ce_loss
                    + (1 - ALPHA) * TEMPERATURE**2 * kd_loss
                )
            else:
                loss = ce_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        print(
            f"  Epoch {epoch + 1}/{EPOCHS} "
            f"| Loss: {avg_loss:.4f}",
            flush=True
        )


def measure_latency(model):
    model.eval()
    sample = torch.randn(1, 1, 28, 28, device=DEVICE)
    measurements = []

    with torch.inference_mode():
        for _ in range(50):
            model(sample)

        for _ in range(5):
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            for _ in range(200):
                model(sample)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            measurements.append(elapsed * 1000 / 200)

    return statistics.median(measurements)


def main():
    print("LAB 07 - TASK 4: KNOWLEDGE DISTILLATION")
    print("-" * 55)
    print("Device:", DEVICE)
    print("Temperature:", TEMPERATURE)
    print("Alpha:", ALPHA)

    # Load trained teacher
    teacher = BaselineCNN().to(DEVICE)
    teacher.load_state_dict(
        torch.load(
            MODELS / "baseline_cnn.pth",
            map_location=DEVICE,
            weights_only=True
        )
    )
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad_(False)

    teacher_params = sum(
        p.numel() for p in teacher.parameters()
    )

    print(f"Teacher parameters: {teacher_params:,}")

    results = []

    # Compare plain student and distilled student
    for name, use_teacher in [
        ("plain_student", False),
        ("distilled_student", True)
    ]:
        print(f"\nTraining: {name}")

        # Same initialization seed for fair comparison
        torch.manual_seed(SEED)
        student = StudentCNN().to(DEVICE)

        student_params = sum(
            p.numel() for p in student.parameters()
        )

        print(f"Student parameters: {student_params:,}")

        train_student(
            student,
            teacher=teacher if use_teacher else None
        )

        accuracy = evaluate(student)
        latency = measure_latency(student)

        path = MODELS / f"{name}.pth"
        torch.save(student.state_dict(), path)

        size_mb = path.stat().st_size / (1024 ** 2)

        print(f"  Test accuracy: {accuracy:.2f}%")
        print(f"  Model size: {size_mb:.3f} MB")
        print(f"  Latency: {latency:.4f} ms")

        results.append({
            "model": name,
            "parameters": student_params,
            "accuracy_percent": accuracy,
            "size_mb": size_mb,
            "latency_ms": latency
        })

    # Teacher reference
    teacher_accuracy = evaluate(teacher)
    teacher_latency = measure_latency(teacher)
    teacher_path = MODELS / "baseline_cnn.pth"

    results.insert(0, {
        "model": "teacher",
        "parameters": teacher_params,
        "accuracy_percent": teacher_accuracy,
        "size_mb": teacher_path.stat().st_size / (1024 ** 2),
        "latency_ms": teacher_latency
    })

    output = RESULTS / "task4_results.csv"

    with open(output, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nDISTILLATION SUMMARY")
    print("-" * 55)

    for row in results:
        print(
            f"{row['model']:20} | "
            f"Accuracy: {row['accuracy_percent']:.2f}% | "
            f"Parameters: {row['parameters']:,} | "
            f"Size: {row['size_mb']:.3f} MB | "
            f"Latency: {row['latency_ms']:.4f} ms"
        )

    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()

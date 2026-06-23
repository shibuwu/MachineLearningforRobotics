data = [
    ("BC Lift baseline",             0.20, "4/20",  "N/A"),
    ("BC PickPlaceCan baseline",     0.00, "0/20",  "0.018"),
    ("BC PickPlaceCan improved",     0.00, "0/20",  "0.0365"),
    ("Diffusion Lift baseline",      0.00, "0/20",  "N/A"),
    ("Diffusion PickPlace baseline", 0.00, "0/20",  "overfit"),
    ("Diffusion PickPlace improved", 0.10, "2/20",  "0.0193"),
]
data.sort(key=lambda x: -x[1])
print("Model                           Success   Successes  Valid Loss")
print("-" * 65)
for name, sr, num, vl in data:
    print(f"{name:<32} {str(int(sr*100))+'%':<10}{num:<11}{vl}")
print()
print("WINNER: Diffusion PickPlaceCan improved - 10% success rate")
print("Best checkpoint: model_epoch_110_best_validation_0.0193.pth")

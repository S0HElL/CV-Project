import yaml

def main():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("Evaluating Depth with config:")
    print(config)

if __name__ == "__main__":
    main()

import argparse
import threading

from robot_commander import config as cfg
from robot_commander.remote_control.agent_client import AgentClient


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true")
    return parser.parse_args()


def _stream_positions(client: AgentClient) -> None:
    for x, y, heading in client.stream_positions():
        print(f"pos  x={x:.3f}  y={y:.3f}  heading={heading:.3f}", flush=True)


def _stream_updates(client: AgentClient) -> None:
    for _camera, rays, cone, _payload in client.stream_agent_updates():
        if rays is not None:
            colliding = [r for r in rays if r[4]]
            nearest = min((abs(r[2] - r[0]) ** 2 + abs(r[3] - r[1]) ** 2) ** 0.5 for r in rays) if rays else float("inf")
            print(f"sensor  rays={len(rays)}  nearest_hit={nearest:.3f}m  collisions={len(colliding)}", flush=True)
        elif cone is not None:
            distance_m, heading = cone
            print(f"sensor  cone  distance={distance_m:.3f}m  heading={heading:.3f}", flush=True)


def main() -> None:
    args = _parse_args()
    connection = cfg.load().connection
    host = connection.simulated_host if args.simulate else connection.host
    client = AgentClient(host=host)

    position_thread = threading.Thread(target=_stream_positions, args=(client,), daemon=True)
    update_thread = threading.Thread(target=_stream_updates, args=(client,), daemon=True)

    position_thread.start()
    update_thread.start()

    try:
        position_thread.join()
        update_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        client.close()


if __name__ == "__main__":
    main()

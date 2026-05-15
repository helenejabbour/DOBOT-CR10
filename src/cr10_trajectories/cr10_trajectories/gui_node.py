#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import tkinter as tk
from tkinter import ttk
import threading


STATUS_COLORS = {
    0: ('blue',   'ℹ '),    # info
    1: ('orange', '⚠ '),   # warning
    2: ('red',    '✗ '),    # error
    3: ('green',  '✓ '),   # success
}


class GUIPublisher(Node):
    def __init__(self):
        super().__init__('gui_publisher')
        self.straight_pub = self.create_publisher(
            Float64MultiArray, 'target_points', 10)
        self.circle_pub = self.create_publisher(
            Float64MultiArray, 'circle_params', 10)

        # Subscribe to status from trajectory_node
        self.create_subscription(
            Float64MultiArray, 'trajectory_status', self.status_callback, 10)

        self._status_callback = None   # set by GUI after launch

    def set_status_callback(self, cb):
        self._status_callback = cb

    def status_callback(self, msg):
        code = int(msg.data[0])
        if self._status_callback:
            self._status_callback(code)

    def publish_straight(self, A, B, speed, elbow_mode_value=0.0):
        msg = Float64MultiArray()
        msg.data = A + B + [speed, float(elbow_mode_value)]
        self.straight_pub.publish(msg)

    def publish_circle(self, center, R, speed, elbow_mode_value=0.0):
        msg = Float64MultiArray()
        msg.data = center + [R, float(speed), float(elbow_mode_value)]
        self.circle_pub.publish(msg)


def launch_gui(node):
    root = tk.Tk()
    root.title('CR10 Trajectory GUI')
    root.resizable(False, False)

    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True, padx=10, pady=10)

    # Shared status bar at the bottom
    status_frame = tk.Frame(root)
    status_frame.pack(fill='x', padx=10, pady=(0, 10))
    status_label = tk.Label(
        status_frame, text='Ready.', anchor='w',
        font=('Helvetica', 10), wraplength=400)
    status_label.pack(fill='x')

    def set_status(text, color='black'):
        status_label.config(text=text, fg=color)

    def node_status_callback(code):
        prefix_color = STATUS_COLORS.get(code, ('black', ''))
        color, prefix = prefix_color
        messages = {
            0: 'Processing...',
            1: 'Warning: some waypoints skipped. Check terminal.',
            2: 'Error: trajectory aborted. Check inputs.',
            3: 'Success: trajectory complete.',
        }
        root.after(0, lambda: set_status(prefix + messages.get(code, ''), color))

    node.set_status_callback(node_status_callback)

    # ------------------------------------------------------------------ #
    #  Tab 1: Straight-line
    # ------------------------------------------------------------------ #
    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text='  Straight-Line A → B  ')

    def row(parent, label, r, defaults=('', '', '')):
        tk.Label(parent, text=label, anchor='w').grid(
            row=r, column=0, padx=8, pady=6, sticky='w')
        entries = []
        for c, d in enumerate(defaults, start=1):
            e = tk.Entry(parent, width=9)
            e.insert(0, d)
            e.grid(row=r, column=c, padx=3)
            entries.append(e)
        return entries

    def decode_elbow_mode(value):
        mapping = {
            'Auto': 0.0,
            'Elbow Up': 1.0,
            'Elbow Down': -1.0,
        }
        return mapping.get(value, 0.0)

    ax, ay, az = row(tab1, 'Point A  (x, y, z) m:', 0)
    bx, by, bz = row(tab1, 'Point B  (x, y, z) m:', 1)

    tk.Label(tab1, text='Speed  (1=slow … 10=fast):', anchor='w').grid(
        row=2, column=0, padx=8, pady=6, sticky='w')
    spd1 = tk.Entry(tab1, width=9); spd1.insert(0, '2')
    spd1.grid(row=2, column=1, padx=3)

    tk.Label(tab1, text='Approach mode:', anchor='w').grid(
        row=2, column=2, padx=8, pady=6, sticky='w')
    elbow_mode_straight = ttk.Combobox(
        tab1,
        values=['Auto', 'Elbow Up', 'Elbow Down'],
        state='readonly',
        width=12,
    )
    elbow_mode_straight.set('Auto')
    elbow_mode_straight.grid(row=2, column=3, padx=3)

    def run_straight():
        try:
            A = [float(ax.get()), float(ay.get()), float(az.get())]
            B = [float(bx.get()), float(by.get()), float(bz.get())]
            speed = float(spd1.get())
            if speed <= 0:
                set_status('✗ Speed must be > 0.', 'red'); return
            elbow_mode = decode_elbow_mode(elbow_mode_straight.get())
            node.publish_straight(A, B, speed, elbow_mode)
            set_status(
                f'ℹ Sent: A={A}  B={B}  speed={speed}  mode={elbow_mode_straight.get()}',
                'blue',
            )
        except ValueError:
            set_status('✗ Invalid input. All fields must be numbers.', 'red')

    tk.Button(tab1, text='▶  Run Straight-Line', command=run_straight,
              bg='#1a6faf', fg='white', width=22,
              font=('Helvetica', 10, 'bold')).grid(
        row=3, column=0, columnspan=4, pady=12)

    # ------------------------------------------------------------------ #
    #  Tab 2: Circular
    # ------------------------------------------------------------------ #
    tab2 = ttk.Frame(notebook)
    notebook.add(tab2, text='  Circular Trajectory  ')

    cx, cy, cz = row(tab2, 'Center  (x, y, z) m:', 0)

    tk.Label(tab2, text='Radius  (m):', anchor='w').grid(
        row=1, column=0, padx=8, pady=6, sticky='w')
    rad = tk.Entry(tab2, width=9)
    rad.grid(row=1, column=1, padx=3)

    tk.Label(tab2, text='Speed  (1=slow … 10=fast):', anchor='w').grid(
        row=2, column=0, padx=8, pady=6, sticky='w')
    spd2 = tk.Entry(tab2, width=9); spd2.insert(0, '2')
    spd2.grid(row=2, column=1, padx=3)

    tk.Label(tab2, text='Approach mode:', anchor='w').grid(
        row=2, column=2, padx=8, pady=6, sticky='w')
    elbow_mode_circle = ttk.Combobox(
        tab2,
        values=['Auto', 'Elbow Up', 'Elbow Down'],
        state='readonly',
        width=12,
    )
    elbow_mode_circle.set('Auto')
    elbow_mode_circle.grid(row=2, column=3, padx=3)

    def run_circle():
        try:
            center = [float(cx.get()), float(cy.get()), float(cz.get())]
            R = float(rad.get())
            speed = float(spd2.get())
            if R <= 0:
                set_status('✗ Radius must be > 0.', 'red'); return
            if speed <= 0:
                set_status('✗ Speed must be > 0.', 'red'); return
            elbow_mode = decode_elbow_mode(elbow_mode_circle.get())
            node.publish_circle(center, R, speed, elbow_mode)
            set_status(
                f'ℹ Sent: center={center}  R={R}m  speed={speed}  mode={elbow_mode_circle.get()} (adaptive points)',
                'blue')
        except ValueError:
            set_status('✗ Invalid input. All fields must be numbers.', 'red')

    tk.Button(tab2, text='▶  Run Circle', command=run_circle,
              bg='#1a6faf', fg='white', width=22,
              font=('Helvetica', 10, 'bold')).grid(
        row=3, column=0, columnspan=4, pady=12)

    # ROS spin in background thread so GUI stays responsive
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    node = GUIPublisher()
    try:
        launch_gui(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

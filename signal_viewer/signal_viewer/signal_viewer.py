import reflex as rx
from rxconfig import config


def my_page():
    return rx.box(
        rx.text('this is something'),
        rx.button('click me')
    )


app = rx.App()
app.add_page(my_page)

from src.interface import gradio_UI



if __name__ == "__main__":
    demo, model = gradio_UI.create_interface()
    demo.launch(
        server_name="0.0.0.0",  # Доступ извне
        server_port=7777,
        share=True,  # Публичная ссылка
        show_error=True,
        debug=True
    )
from dash import html, callback, Input, Output

def create_callback_for_selectbox(input_id, output_id):
    """
    Function that creates a callback for updating a component
    in select box
    """
    @callback(
        Output(output_id, 'value'),
        Input(input_id, 'value')
    )
    def update_output(value):
        return f'You have entered: {value}'

def create_callback_for_selecting(input_id, output_id, selectbox_id):
    """
    Function to get the value when selected
    """
    @callback(
        Output(output_id, 'value'),
        Output(selectbox_id, 'value'),
        Input(input_id, 'value')
    )
    def update_value(value):
        return value, False
def Selectbox(input_id, output_id):
    return html.Div(
        children=[
            
        ],
        className="relative"
    )

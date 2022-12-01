

def training_args(hyper_params, batch_setting):
	
	all_kwags = {}

	# Define the generator model 
    modelG_kwags = {'type': None, 'args': {}}
    modelG_kwags['type'] = 'UNet'
    modelG_kwags['args']['n_channels'] = 2
    modelG_kwags['args']['n_classes'] = 1
    modelG_kwags['args']['depth'] = 2
    all_kwags['modelG_kwags'] = modelG_kwags

    # Define the discriminator model 
    modelD_kwags = {'type': None, 'args': {}}
    modelD_kwags['type'] = 'ResNet18'
    modelD_kwags['args']['input_channel'] = 1
    modelD_kwags['args']['output_dim'] = 2
    all_kwags['modelD_kwags'] = modelD_kwags
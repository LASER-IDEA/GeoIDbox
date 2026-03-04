import xarray as xr

# Load the dataset
ds = xr.open_dataset('/data/home/huxiao/workspace/geobox/GeoIDbox/data/era5_shenzhen_complete.nc')

# Print the variables and their attributes
print("Dataset Variables:")
for var_name in ds.variables:
    var = ds[var_name]
    print(f"\nVariable: {var_name}")
    print(f"  Dimensions: {var.dims}")
    print(f"  Shape: {var.shape}")
    print(f"  Attributes: {var.attrs}")

# Check for 'z' (geopotential) or 'orography' explicitly
if 'z' in ds.variables:
    print("\nFound variable 'z' (geopotential).")
else:
    print("\nVariable 'z' (geopotential) NOT found.")

if 'orography' in ds.variables:
    print("\nFound variable 'orography'.")
else:
    print("\nVariable 'orography' NOT found.")

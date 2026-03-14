import logging

from fastmcp import FastMCP
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(levelname)s] - %(message)s'
)

log = logging.getLogger(__name__)

# 1. Initialize the FastMCP server
mcp = FastMCP("Supply Chain Agent")

# 2. In-Memory Mock Databases (Our State)
ORDERS = {
    "9942": {
        "product": "Premium Widgets", 
        "quantity": 150, 
        "destination_city": "New York",
        "status": "Pending"
    }
}

INVENTORY = {
    "Premium Widgets": {
        "Chicago": {"stock": 100},
        "Atlanta": {"stock": 80},
        "Dallas": {"stock": 200}
    }
}

# Cost per item to ship from a warehouse to a destination
SHIPPING_RATES = {
    "Chicago": {"New York": 2.50},
    "Atlanta": {"New York": 3.00},
    "Dallas": {"New York": 5.00}
}

# 3. Pydantic Models for structured inputs
class WarehouseAllocation(BaseModel):
    warehouse_city: str = Field(description="The city where the warehouse is located")
    quantity: int = Field(description="The number of units to ship from this warehouse")

# 4. Tool and Prompt Definitions

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# (Assuming mcp is already initialized as in the previous script)

@mcp.prompt()
def fulfill_order(order_id: str) -> str:
    """Generate the exact instruction set for fulfilling an order."""
    log.info(f"Returning prompt for order id: {order_id}")
    return f"""
    You are an expert Logistics AI. Your task is to fulfill Order #{order_id}.
    
    Please follow these exact steps:
    1. Fetch the details for Order #{order_id}.
    3. Check the inventory for the required product.
    3. Calculate shipping costs using your tools.
    4. Execute the most efficient shipment.
    5. Provide a step-by-step breakdown of your mathematical reasoning.
    """


@mcp.tool()
def get_order_details(order_id: str) -> dict:
    """
    Get the details of an order including product name, quantity required, and destination.
    """
    log.info(f"Returning order details for order id: {order_id}")
    if order_id not in ORDERS:
        return {"error": f"Order {order_id} not found."}
    
    return ORDERS[order_id]

@mcp.tool()
def check_inventory(product_name: str) -> dict:
    """
    Check the current inventory levels for a specific product across all warehouses.
    """
    log.info(f"Checking inventory for product name {product_name}")
    if product_name not in INVENTORY:
        return {"error": f"Product '{product_name}' not found in inventory."}
        
    return INVENTORY[product_name]

@mcp.tool()
def get_shipping_rates(warehouse_city: str, destination_city: str, quantity: int) -> str:
    """
    Calculate the total shipping cost to send a specific quantity of items from a warehouse to a destination.
    """
    log.info(f"Fetching shipping rates for {warehouse_city=} {destination_city=} {quantity=}")
    rates = SHIPPING_RATES.get(warehouse_city, {})
    
    if destination_city not in rates:
        return f"Error: No shipping routes available from {warehouse_city} to {destination_city}."
        
    cost_per_item = rates[destination_city]
    total_cost = cost_per_item * quantity
    
    return f"Total cost to ship {quantity} items from {warehouse_city} to {destination_city} is ${total_cost:.2f} (${cost_per_item}/item)."

@mcp.tool()
def create_shipment(order_id: str, allocations: list[WarehouseAllocation]) -> str:
    """
    Execute a shipment to fulfill an order by allocating inventory from specified warehouses.
    Requires a list of allocations detailing how many items come from which warehouse.
    """
    log.info(f"Creating shipment for {order_id=} {allocations=}")
    if order_id not in ORDERS:
        return f"Error: Order {order_id} not found."
        
    order = ORDERS[order_id]
    product = order["product"]
    
    if order["status"] == "Shipped":
        return f"Error: Order {order_id} has already been shipped."
    
    # Validate total quantity
    total_allocated = sum(alloc.quantity for alloc in allocations)
    if total_allocated != order["quantity"]:
        return f"Error: You allocated {total_allocated} items, but the order requires {order['quantity']}."
        
    # Validate inventory availability (Check before deducting)
    for alloc in allocations:
        stock = INVENTORY.get(product, {}).get(alloc.warehouse_city, {}).get("stock", 0)
        if alloc.quantity > stock:
            return f"Error: Insufficient stock in {alloc.warehouse_city}. Requested {alloc.quantity}, but only {stock} available."
            
    # Execute the deduction (Modify State)
    for alloc in allocations:
        INVENTORY[product][alloc.warehouse_city]["stock"] -= alloc.quantity
        
    ORDERS[order_id]["status"] = "Shipped"
    
    return f"Success! Order {order_id} fulfilled. Inventory updated successfully."

# 5. Entry Point
if __name__ == "__main__":
    mcp.run(transport='http', port=8000)


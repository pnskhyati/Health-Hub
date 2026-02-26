import React, { useState } from "react";
import { useCart } from "../Cart/CartContext";
import { Dialog, DialogActions, DialogContent, DialogTitle, Button, Rating, Divider, Typography, IconButton } from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import Alert from "./Alert";
import "./SingleProduct.css";

export default function SingleProduct({ product, openDialog, closeDialog }) {
  const { addToCart } = useCart();
  const [showAlert, setShowAlert] = useState(false);

  const handleAddToCart = (product) => {
    addToCart(product);
    setShowAlert(true);
    setTimeout(() => setShowAlert(false), 3000); // Hide alert after 3 seconds
  };

  return (
    <>
      <Dialog
        open={openDialog}
        onClose={closeDialog}
        fullWidth
        maxWidth="md"
        className="product-dialog"
      >
        <DialogTitle className="dialog-title">
          <div className="dialog-title-content">
            <Typography variant="h6">{product.MedicineName}</Typography>
            <IconButton onClick={closeDialog} className="close-dialog">
              <CloseIcon />
            </IconButton>
          </div>
        </DialogTitle>

        <DialogContent className="dialog-content">
          {showAlert && <Alert message={`${product.MedicineName} added to Cart!`} />}
          <div className="product-details-container">
            {/* Left Section: Product Image */}
            <div className="product-image-container">
              <img
                src={product.image}
                alt={product.MedicineName}
                className="product-img"
              />
            </div>

            {/* Right Section: Product Details */}
            <div className="product-info">
              <Typography className="product-price" variant="h6">
                Price: ₹{product.MRP}
              </Typography>
              <div className="product-rating">
                <Rating value={product.Ratings || 4.5} precision={0.5} readOnly />
                <Typography>({product.Reviews || "0"} Reviews)</Typography>
              </div>
              <Divider style={{ margin: "10px 0" }} />
              <Typography className="product-description">
                {product.description}
              </Typography>

              <Typography variant="subtitle1" gutterBottom>
                <strong>Key Benefits:</strong>
              </Typography>
              <ul className="key-benefits-list">
                {product.keyBenefits?.map((benefit, index) => (
                  <li key={index}>{benefit}</li>
                ))}
              </ul>

              <Typography variant="subtitle1" gutterBottom>
                <strong>How to Use:</strong>
              </Typography>
              <Typography>{product.howToUse}</Typography>

              <Typography variant="subtitle1" gutterBottom>
                <strong>Precautions:</strong>
              </Typography>
              <Typography>{product.precautions}</Typography>
            </div>
          </div>

          {/* Additional Information */}
          <div className="additional-info">
            <Typography variant="subtitle1" gutterBottom>
              <strong>Ingredients:</strong>
            </Typography>
            <Typography>{product.ingredients}</Typography>

            <Typography variant="subtitle1" gutterBottom>
              <strong>Quantity:</strong>
            </Typography>
            <Typography>{product.quantity}</Typography>
          </div>
        </DialogContent>

        <DialogActions className="dialog-actions">
          <Button onClick={closeDialog} color="secondary">
            Close
          </Button>
          <Button
            onClick={() => handleAddToCart(product)}
            variant="contained"
            className="add-to-cart-button"
          >
            Add to Cart
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

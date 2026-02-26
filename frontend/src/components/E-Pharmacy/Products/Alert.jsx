
import React from 'react';
import { Alert as BootstrapAlert } from 'react-bootstrap';
import './SingleProduct.css'; // If you want to apply some custom styles to the alert

const Alert = (props) => {
  return (
    <BootstrapAlert variant="success" className="fixed-top alert-notification">
      {props.message}
    </BootstrapAlert>
  );
};

export default Alert;

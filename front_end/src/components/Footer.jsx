import { Link } from "react-router-dom";

const Footer = () => {
  return (
    <footer className="bg-gray-800 text-white py-8 mt-12">
      <div className="container mx-auto px-4">
        <div className="text-center">
          <p className="text-gray-400">
            © 2026 Fat Predictor. All rights reserved.
          </p>
          <p className="text-gray-500 text-sm mt-2">
            This tool provides estimates only. Consult healthcare professionals
            for medical advice.
          </p>
          <p className="text-gray-500 text-sm mt-2">
            This app is a hosted portfolio project for demonstration purposes.
          </p>
          <div className="mt-6">
            <ul className="inline-flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-sm text-gray-300 list-disc list-inside">
              <li>
                <Link
                  to="/about"
                  className="hover:text-white transition-colors"
                >
                  About
                </Link>
              </li>
              <li>
                <Link
                  to="/privacy-policy"
                  className="hover:text-white transition-colors"
                >
                  Privacy Policy
                </Link>
              </li>
              <li>
                <Link
                  to="/terms-of-use"
                  className="hover:text-white transition-colors"
                >
                  Terms of Use
                </Link>
              </li>
              <li>
                <a
                  href="mailto:support@example.com"
                  className="hover:text-white transition-colors"
                >
                  Contact
                </a>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;

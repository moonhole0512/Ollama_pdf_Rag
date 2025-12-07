import React from 'react';
import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Typography,
  Box,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

interface DocumentSource {
  page_content: string;
  metadata: {
    page: number;
    source: string;
  };
}

interface SourceAccordionProps {
  sources: DocumentSource[];
}

const SourceAccordion: React.FC<SourceAccordionProps> = ({ sources }) => {
  if (!sources || sources.length === 0) {
    return null;
  }

  return (
    <Accordion sx={{ mt: 2, boxShadow: 'none', border: '1px solid rgba(0, 0, 0, 0.12)' }}>
      <AccordionSummary
        expandIcon={<ExpandMoreIcon />}
        aria-controls="panel1a-content"
        id="panel1a-header"
      >
        <Typography variant="body2" sx={{ fontWeight: 'medium' }}>
          View Sources ({sources.length})
        </Typography>
      </AccordionSummary>
      <AccordionDetails sx={{ maxHeight: '200px', overflowY: 'auto' }}>
        {sources.map((source, index) => (
          <Box
            key={index}
            sx={{
              mb: 2,
              p: 1,
              border: '1px solid #e0e0e0',
              borderRadius: '4px',
              backgroundColor: '#fafafa',
            }}
          >
            <Typography variant="caption" color="textSecondary">
              Source {index + 1} (Page: {source.metadata.page + 1})
            </Typography>
            <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
              {source.page_content}
            </Typography>
          </Box>
        ))}
      </AccordionDetails>
    </Accordion>
  );
};

export default SourceAccordion;

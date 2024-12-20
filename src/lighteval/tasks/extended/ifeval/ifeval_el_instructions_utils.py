# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adaptation of the IFEVAL instructions_utils.py to a greek counterpart

"""Utility library of instructions."""

import functools
import random
import re

import nltk


def download_nltk_resources():
    """Download 'punkt' if not already installed"""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


download_nltk_resources()

WORD_LIST = [
    "δυτικός",
    "πρόταση",
    "σήμα",
    "απόρριψη",
    "σημείο",
    "αντίθετο",
    "κάτω μέρος",
    "πατάτα",
    "διοίκηση",
    "εργασία",
    "καλωσόρισμα",
    "πρωί",
    "καλός",
    "πρακτορείο",
    "κύριος",
    "ευχή",
    "ευθύνη",
    "τύπος",
    "πρόβλημα",
    "πρόεδρος",
    "κλέβω",
    "βούρτσα",
    "διαβάζω",
    "τύπος",
    "χτυπάω",
    "προπονητής",
    "ανάπτυξη",
    "κλειδαριά",
    "κόκκαλο",
    "περίπτωση",
    "ίσος",
    "άνετος",
    "περιοχή",
    "αντικατάσταση",
    "απόδοση",
    "σύντροφος",
    "περπάτημα",
    "ιατρική",
    "ταινία",
    "πράγμα",
    "βράχος",
    "χτύπημα",
    "σύνολο",
    "ανταγωνισμός",
    "ευκολία",
    "νότος",
    "ίδρυμα",
    "συγκεντρώνω",
    "στάθμευση",
    "κόσμος",
    "άφθονος",
    "ανάσα",
    "αξίωση",
    "αλκοόλ",
    "εμπόριο",
    "αγαπητός",
    "επισήμανση",
    "οδός",
    "θέμα",
    "απόφαση",
    "χάος",
    "συμφωνία",
    "στούντιο",
    "προπονητής",
    "βοηθάω",
    "εγκέφαλος",
    "φτερό",
    "στυλ",
    "ιδιωτικός",
    "κορυφή",
    "καφέ",
    "πόδι",
    "αγοράζω",
    "διαδικασία",
    "μέθοδος",
    "ταχύτητα",
    "υψηλός",
    "εταιρεία",
    "πολύτιμος",
    "πίτα",
    "αναλυτής",
    "συνεδρία",
    "μοτίβο",
    "περιοχή",
    "ευχαρίστηση",
    "δείπνο",
    "κολύμβηση",
    "αστείο",
    "παραγγελία",
    "πιάτο",
    "τμήμα",
    "κινητήρας",
    "κύτταρο",
    "ξοδεύω",
    "ντουλάπι",
    "διαφορά",
    "δύναμη",
    "εξέταση",
    "μηχανή",
    "άλογο",
    "διάσταση",
    "πληρώνω",
    "δάχτυλο ποδιού",
    "καμπύλη",
    "λογοτεχνία",
    "ενοχλώ",
    "φωτιά",
    "δυνατότητα",
    "συζήτηση",
    "δραστηριότητα",
    "πέρασμα",
    "γεια",
    "κύκλος",
    "υπόβαθρο",
    "ήσυχος",
    "συγγραφέας",
    "αποτέλεσμα",
    "ηθοποιός",
    "σελίδα",
    "ποδήλατο",
    "λάθος",
    "λαιμός",
    "επίθεση",
    "χαρακτήρας",
    "τηλέφωνο",
    "τσάι",
    "αύξηση",
    "αποτέλεσμα",
    "αρχείο",
    "συγκεκριμένος",
    "επιθεωρητής",
    "εσωτερικός",
    "δυνατότητα",
    "προσωπικό",
    "κτίριο",
    "εργοδότης",
    "παπούτσι",
    "χέρι",
    "κατεύθυνση",
    "κήπος",
    "αγορά",
    "συνέντευξη",
    "μελέτη",
    "αναγνώριση",
    "μέλος",
    "πνευματικός",
    "φούρνος",
    "σάντουιτς",
    "παράξενος",
    "επιβάτης",
    "συγκεκριμένος",
    "απάντηση",
    "αντίδραση",
    "μέγεθος",
    "παραλλαγή",
    "ένα",
    "ακυρώνω",
    "γλυκό",
    "έξοδος",
    "φιλοξενούμενος",
    "συνθήκη",
    "πετάω",
    "τιμή",
    "αδυναμία",
    "μετατρέπω",
    "ξενοδοχείο",
    "μεγάλος",
    "στόμα",
    "μυαλό",
    "τραγούδι",
    "ζάχαρη",
    "ύποπτος",
    "τηλέφωνο",
    "αυτί",
    "στέγη",
    "βάφω",
    "ψυγείο",
    "οργάνωση",
    "επιτροπή",
    "ανταμοιβή",
    "μηχανική",
    "ημέρα",
    "κατοχή",
    "πλήρωμα",
    "μπαρ",
    "δρόμος",
    "περιγραφή",
    "γιορτή",
    "σκορ",
    "σημάδι",
    "γράμμα",
    "ντους",
    "πρόταση",
    "κύριος",
    "τύχη",
    "εθνικός",
    "πρόοδος",
    "αίθουσα",
    "εγκεφαλικό",
    "θεωρία",
    "προσφορά",
    "ιστορία",
    "φόρος",
    "ορισμός",
    "ιστορία",
    "βόλτα",
    "μεσαίος",
    "άνοιγμα",
    "ποτήρι",
    "ανελκυστήρας",
    "στομάχι",
    "ερώτηση",
    "ικανότητα",
    "ηγέτης",
    "χωριό",
    "υπολογιστής",
    "πόλη",
    "μεγάλος",
    "εμπιστοσύνη",
    "κερί",
    "ιερέας",
    "σύσταση",
    "σημείο",
    "απαραίτητος",
    "σώμα",
    "γραφείο",
    "μυστικό",
    "τρόμος",
    "θόρυβος",
    "κουλτούρα",
    "προειδοποίηση",
    "νερό",
    "στρογγυλός",
    "δίαιτα",
    "λουλούδι",
    "λεωφορείο",
    "σκληρός",
    "άδεια",
    "εβδομάδα",
    "παρακινητικός",
    "σύνδεση",
    "κακοποίηση",
    "ύψος",
    "σώζω",
    "γωνία",
    "σύνορο",
    "στρες",
    "οδηγώ",
    "σταματάω",
    "σκίζω",
    "γεύμα",
    "ακούω",
    "σύγχυση",
    "φίλη",
    "ζωντανός",
    "σχέση",
    "σημασία",
    "σχέδιο",
    "δημιουργικός",
    "ατμόσφαιρα",
    "κατηγορώ",
    "προσκαλώ",
    "στέγαση",
    "χαρτί",
    "ποτό",
    "ρολάρω",
    "ασήμι",
    "μεθυσμένος",
    "ηλικία",
    "ζημιά",
    "καπνός",
    "περιβάλλον",
    "πακέτο",
    "αποταμιεύσεις",
    "επιρροή",
    "τουρίστας",
    "βροχή",
    "ταχυδρομείο",
    "σήμα",
    "γιαγιά",
    "τρέχω",
    "κέρδος",
    "σπρώχνω",
    "υπάλληλος",
    "τελικός",
    "κρασί",
    "κολυμπάω",
    "παύση",
    "πράγματα",
    "τραγουδιστής",
    "κηδεία",
    "μέσος όρος",
    "πηγή",
    "σκηνή",
    "παράδοση",
    "προσωπικός",
    "χιόνι",
    "κανείς",
    "απόσταση",
    "είδος",
    "ευαίσθητος",
    "ζώο",
    "κύριος",
    "διαπραγμάτευση",
    "κλικ",
    "διάθεση",
    "περίοδος",
    "άφιξη",
    "έκφραση",
    "διακοπές",
    "επαναλαμβάνω",
    "σκόνη",
    "ντουλάπα",
    "χρυσός",
    "κακός",
    "πανί",
    "συνδυασμός",
    "ρούχα",
    "έμφαση",
    "καθήκον",
    "μαύρος",
    "βήμα",
    "σχολείο",
    "πηδώ",
    "έγγραφο",
    "επαγγελματίας",
    "χείλος",
    "χημικός",
    "μπροστινός",
    "ξυπνάω",
    "ενώ",
    "μέσα",
    "ρολόι",
    "σειρά",
    "θέμα",
    "ποινή",
    "ισορροπία",
    "πιθανός",
    "ενήλικας",
    "παράμερα",
    "δείγμα",
    "έφεση",
    "γάμος",
    "βάθος",
    "βασιλιάς",
    "βραβείο",
    "σύζυγος",
    "χτύπημα",
    "ιστότοπος",
    "στρατόπεδο",
    "μουσική",
    "ασφαλής",
    "δώρο",
    "λάθος",
    "μαντεύω",
    "πράξη",
    "ντροπή",
    "δράμα",
    "κεφάλαιο",
    "εξέταση",
    "ηλίθιος",
    "εγγραφή",
    "ήχος",
    "ταλαντεύομαι",
    "μυθιστόρημα",
    "ελάχιστο",
    "αναλογία",
    "μηχανή",
    "σχήμα",
    "μολύβι",
    "λειτουργία",
    "μισθός",
    "σύννεφο",
    "υπόθεση",
    "χτυπάω",
    "κεφάλαιο",
    "στάδιο",
    "ποσότητα",
    "πρόσβαση",
    "στρατός",
    "αλυσίδα",
    "κυκλοφορία",
    "κλωτσιά",
    "ανάλυση",
    "αεροδρόμιο",
    "χρόνος",
    "διακοπές",
    "φιλοσοφία",
    "μπάλα",
    "στήθος",
    "ευχαριστώ",
    "τόπος",
    "βουνό",
    "διαφήμιση",
    "κόκκινος",
    "παρελθόν",
    "ενοίκιο",
    "επιστροφή",
    "περιοδεία",
    "σπίτι",
    "κατασκευή",
    "δίκτυο",
    "ντόπιος",
    "πόλεμος",
    "σχήμα",
    "αμοιβή",
    "ψεκασμός",
    "χρήστης",
    "βρωμιά",
    "βολή",
    "εργασία",
    "ραβδί",
    "φίλος",
    "λογισμικό",
    "προαγωγή",
    "αλληλεπίδραση",
    "περιβάλλω",
    "μπλοκ",
    "σκοπός",
    "πρακτική",
    "σύγκρουση",
    "ρουτίνα",
    "απαίτηση",
    "μπόνους",
    "τρύπα",
    "κράτος",
    "νεότερος",
    "γλυκός",
    "συλλαμβάνω",
    "δάκρυ",
    "διπλώνω",
    "τοίχος",
    "συντάκτης",
    "ζωή",
    "θέση",
    "λίβρα",
    "σεβασμός",
    "μπάνιο",
    "παλτό",
    "σενάριο",
    "δουλειά",
    "διδάσκω",
    "γέννηση",
    "θέα",
    "επιλύω",
    "θέμα",
    "υπάλληλος",
    "αμφιβολία",
    "αγορά",
    "εκπαίδευση",
    "υπηρετώ",
    "αναρρώνω",
    "τόνος",
    "βλάβη",
    "χάνω",
    "ένωση",
    "κατανόηση",
    "αγελάδα",
    "ποτάμι",
    "σύνδεσμος",
    "έννοια",
    "εκπαίδευση",
    "συνταγή",
    "σχέση",
    "εφεδρεία",
    "κατάθλιψη",
    "απόδειξη",
    "μαλλιά",
    "έσοδα",
    "ανεξάρτητος",
    "ανελκυστήρας",
    "ανάθεση",
    "προσωρινός",
    "ποσό",
    "απώλεια",
    "άκρη",
    "διαδρομή",
    "ελέγχω",
    "σχοινί",
    "εκτίμηση",
    "ρύπανση",
    "σταθερός",
    "μήνυμα",
    "παράδοση",
    "προοπτική",
    "καθρέφτης",
    "βοηθός",
    "εκπρόσωπος",
    "μάρτυρας",
    "φύση",
    "δικαστής",
    "φρούτο",
    "συμβουλή",
    "διάβολος",
    "πόλη",
    "έκτακτη ανάγκη",
    "άνω",
    "σταγόνα",
    "παραμένω",
    "άνθρωπος",
    "λαιμός",
    "ομιλητής",
    "δίκτυο",
    "τραγουδάω",
    "αντιστέκομαι",
    "λίγκα",
    "ταξίδι",
    "υπογραφή",
    "δικηγόρος",
    "σημασία",
    "αέριο",
    "επιλογή",
    "μηχανικός",
    "επιτυχία",
    "μέρος",
    "εξωτερικός",
    "εργάτης",
    "απλός",
    "τέταρτο",
    "φοιτητής",
    "καρδιά",
    "περνάω",
    "παρά",
    "μετατόπιση",
    "τραχύς",
    "κυρία",
    "γρασίδι",
    "κοινότητα",
    "γκαράζ",
    "νεότητα",
    "πρότυπο",
    "φούστα",
    "υπόσχεση",
    "τυφλός",
    "τηλεόραση",
    "ασθένεια",
    "επιτροπή",
    "θετικός",
    "ενέργεια",
    "ήρεμος",
    "παρουσία",
    "μελωδία",
    "βάση",
    "προτίμηση",
    "κεφάλι",
    "κοινός",
    "κόβω",
    "κάπου",
    "παρουσίαση",
    "τρέχων",
    "σκέψη",
    "επανάσταση",
    "προσπάθεια",
    "αφέντης",
    "εφαρμόζω",
    "δημοκρατία",
    "πάτωμα",
    "αρχή",
    "ξένος",
    "ώμος",
    "βαθμός",
    "κουμπί",
    "τένις",
    "αστυνομία",
    "συλλογή",
    "λογαριασμός",
    "μητρώο",
    "γάντι",
    "διαιρώ",
    "καθηγητής",
    "καρέκλα",
    "προτεραιότητα",
    "συνδυάζω",
    "ειρήνη",
    "επέκταση",
    "ίσως",
    "βράδυ",
    "πλαίσιο",
    "αδελφή",
    "κύμα",
    "κώδικας",
    "εφαρμογή",
    "ποντίκι",
    "αγώνας",
    "πάγκος",
    "μπουκάλι",
    "μισό",
    "μάγουλο",
    "επίλυση",
    "πίσω",
    "γνώση",
    "κάνω",
    "συζήτηση",
    "βίδα",
    "μήκος",
    "ατύχημα",
    "μάχη",
    "φόρεμα",
    "γόνατο",
    "κορμός",
    "πακέτο",
    "αυτό",
    "στροφή",
    "ακοή",
    "εφημερίδα",
    "στρώμα",
    "πλούτος",
    "προφίλ",
    "φαντασία",
    "απάντηση",
    "Σαββατοκύριακο",
    "δάσκαλος",
    "εμφάνιση",
    "συνάντηση",
    "ποδήλατο",
    "άνοδος",
    "ζώνη",
    "συντριβή",
    "μπολ",
    "ισοδύναμο",
    "υποστήριξη",
    "εικόνα",
    "ποίημα",
    "κίνδυνος",
    "ενθουσιασμός",
    "απομακρυσμένος",
    "γραμματέας",
    "δημόσιος",
    "παράγω",
    "αεροπλάνο",
    "οθόνη",
    "χρήματα",
    "άμμος",
    "κατάσταση",
    "γροθιά",
    "πελάτης",
    "τίτλος",
    "κουνάω",
    "υποθήκη",
    "επιλογή",
    "αριθμός",
    "σκάω",
    "παράθυρο",
    "έκταση",
    "τίποτα",
    "εμπειρία",
    "γνώμη",
    "αναχώρηση",
    "χορός",
    "ένδειξη",
    "αγόρι",
    "υλικό",
    "συγκρότημα",
    "ηγέτης",
    "ήλιος",
    "όμορφος",
    "μυς",
    "αγρότης",
    "ποικιλία",
    "λίπος",
    "λαβή",
    "διευθυντής",
    "ευκαιρία",
    "ημερολόγιο",
    "έξω",
    "βήμα",
    "λουτρό",
    "ψάρι",
    "συνέπεια",
    "βάζω",
    "ιδιοκτήτης",
    "πηγαίνω",
    "γιατρός",
    "πληροφορίες",
    "μοιράζομαι",
    "πληγώνω",
    "προστασία",
    "καριέρα",
    "χρηματοδότηση",
    "δύναμη",
    "γκολφ",
    "σκουπίδια",
    "πλευρά",
    "παιδί",
    "φαγητό",
    "μπότα",
    "γάλα",
    "απαντώ",
    "αντικείμενο",
    "πραγματικότητα",
    "ωμός",
    "δαχτυλίδι",
    "εμπορικό κέντρο",
    "ένας",
    "επίδραση",
    "περιοχή",
    "ειδήσεις",
    "διεθνής",
    "σειρά",
    "εντυπωσιάζω",
    "μητέρα",
    "καταφύγιο",
    "απεργία",
    "δάνειο",
    "μήνας",
    "θέση",
    "οτιδήποτε",
    "ψυχαγωγία",
    "γραβάτα",
    "καταστρέφω",
    "άνεση",
    "γη",
    "καταιγίδα",
    "ποσοστό",
    "βοήθεια",
    "προϋπολογισμός",
    "δύναμη",
    "αρχή",
    "ύπνος",
    "άλλος",
    "νέος",
    "μονάδα",
    "γεμίζω",
    "αποθηκεύω",
    "επιθυμία",
    "κρύβω",
    "αξία",
    "φλιτζάνι",
    "συντήρηση",
    "νοσοκόμα",
    "λειτουργία",
    "πύργος",
    "ρόλος",
    "τάξη",
    "κάμερα",
    "βάση δεδομένων",
    "πανικός",
    "έθνος",
    "καλάθι",
    "πάγος",
    "τέχνη",
    "πνεύμα",
    "διάγραμμα",
    "ανταλλαγή",
    "ανατροφοδότηση",
    "δήλωση",
    "φήμη",
    "αναζήτηση",
    "κυνήγι",
    "άσκηση",
    "κακός",
    "ειδοποίηση",
    "άντρας",
    "αυλή",
    "ετήσιος",
    "κολάρο",
    "ημερομηνία",
    "πλατφόρμα",
    "φυτό",
    "τύχη",
    "πάθος",
    "φιλία",
    "διαδίδω",
    "καρκίνος",
    "εισιτήριο",
    "στάση",
    "νησί",
    "ενεργός",
    "αντικείμενο",
    "υπηρεσία",
    "αγοραστής",
    "δάγκωμα",
    "κάρτα",
    "πρόσωπο",
    "μπριζόλα",
    "πρόταση",
    "ασθενής",
    "θερμότητα",
    "κανόνας",
    "κάτοικος",
    "ευρύς",
    "πολιτική",
    "δύση",
    "μαχαίρι",
    "ειδικός",
    "κορίτσι",
    "σχεδιασμός",
    "αλάτι",
    "μπέιζμπολ",
    "αρπάζω",
    "επιθεώρηση",
    "ξάδερφος",
    "ζευγάρι",
    "περιοδικό",
    "μαγειρεύω",
    "εξαρτώμενος",
    "ασφάλεια",
    "κοτόπουλο",
    "έκδοση",
    "νόμισμα",
    "σκάλα",
    "σχέδιο",
    "κουζίνα",
    "απασχόληση",
    "τοπικός",
    "προσοχή",
    "διευθυντής",
    "γεγονός",
    "κάλυμμα",
    "λυπημένος",
    "φύλακας",
    "σχετικός",
    "νομός",
    "ποσοστό",
    "μεσημεριανό",
    "πρόγραμμα",
    "πρωτοβουλία",
    "γρανάζι",
    "γέφυρα",
    "στήθος",
    "ομιλία",
    "πιάτο",
    "εγγύηση",
    "μπύρα",
    "όχημα",
    "υποδοχή",
    "γυναίκα",
    "ουσία",
    "αντίγραφο",
    "διάλεξη",
    "πλεονέκτημα",
    "πάρκο",
    "κρύο",
    "θάνατος",
    "μείγμα",
    "κρατώ",
    "κλίμακα",
    "αύριο",
    "αίμα",
    "αίτημα",
    "πράσινος",
    "μπισκότο",
    "εκκλησία",
    "λωρίδα",
    "για πάντα",
    "πέρα από",
    "χρέος",
    "επιχειρώ",
    "πλένω",
    "ακολουθώντας",
    "αισθάνομαι",
    "μέγιστο",
    "τομέας",
    "θάλασσα",
    "ιδιοκτησία",
    "οικονομία",
    "μενού",
    "παγκάκι",
    "προσπαθώ",
    "γλώσσα",
    "αρχίζω",
    "τηλεφώνημα",
    "στέρεος",
    "διεύθυνση",
    "εισόδημα",
    "πόδι",
    "ανώτερος",
    "μέλι",
    "λίγοι",
    "μείγμα",
    "μετρητά",
    "μπακάλικο",
    "σύνδεσμος",
    "χάρτης",
    "φόρμα",
    "παράγοντας",
    "κατσαρόλα",
    "μοντέλο",
    "συγγραφέας",
    "αγρόκτημα",
    "χειμώνας",
    "ικανότητα",
    "οπουδήποτε",
    "γενέθλια",
    "πολιτική",
    "απελευθέρωση",
    "σύζυγος",
    "εργαστήριο",
    "βιάζομαι",
    "ταχυδρομείο",
    "εξοπλισμός",
    "νεροχύτης",
    "ζευγάρι",
    "οδηγός",
    "σκέψη",
    "δέρμα",
    "μπλε",
    "βάρκα",
    "πώληση",
    "τούβλο",
    "δύο",
    "ταΐζω",
    "τετράγωνο",
    "τελεία",
    "βιάζομαι",
    "όνειρο",
    "τοποθεσία",
    "απόγευμα",
    "κατασκευαστής",
    "έλεγχος",
    "περίσταση",
    "πρόβλημα",
    "εισαγωγή",
    "συμβουλή",
    "στοίχημα",
    "τρώω",
    "σκοτώνω",
    "κατηγορία",
    "τρόπος",
    "γραφείο",
    "κτηματική",
    "περηφάνια",
    "επίγνωση",
    "ολίσθηση",
    "ρωγμή",
    "πελάτης",
    "νύχι",
    "πυροβολώ",
    "συνδρομή",
    "μαλακός",
    "οποιοσδήποτε",
    "διαδίκτυο",
    "επίσημος",
    "άτομο",
    "πίτσα",
    "ενδιαφέρον",
    "τσάντα",
    "ξόρκι",
    "επάγγελμα",
    "βασίλισσα",
    "συμφωνία",
    "πόρος",
    "πλοίο",
    "τύπος",
    "σοκολάτα",
    "κοινός",
    "πάνω",
    "επάνω",
    "αυτοκίνητο",
    "θέρετρο",
    "στο εξωτερικό",
    "έμπορος",
    "συνάδελφος",
    "δάχτυλο",
    "χειρουργική επέμβαση",
    "σχόλιο",
    "ομάδα",
    "λεπτομέρεια",
    "τρελός",
    "μονοπάτι",
    "παραμύθι",
    "αρχικός",
    "χέρι",
    "ραδιόφωνο",
    "απαίτηση",
    "μονός",
    "σχεδιάζω",
    "κίτρινος",
    "διαγωνισμός",
    "κομμάτι",
    "προσφορά",
    "τραβάω",
    "εμπορικός",
    "πουκάμισο",
    "συνεισφορά",
    "κρέμα",
    "κανάλι",
    "κοστούμι",
    "πειθαρχία",
    "οδηγία",
    "συναυλία",
    "ομιλία",
    "χαμηλός",
    "αποτελεσματικός",
    "κρεμάω",
    "γρατζουνιά",
    "βιομηχανία",
    "πρωινό",
    "ξαπλώνω",
    "συνδέομαι",
    "μέταλλο",
    "υπνοδωμάτιο",
    "λεπτό",
    "προϊόν",
    "ξεκούραση",
    "θερμοκρασία",
    "πολλοί",
    "δίνω",
    "επιχείρημα",
    "εκτύπωση",
    "μοβ",
    "γελώ",
    "υγεία",
    "πίστωση",
    "επένδυση",
    "πουλάω",
    "ρύθμιση",
    "μάθημα",
    "αυγό",
    "μεσαίος",
    "γάμος",
    "επίπεδο",
    "απόδειξη",
    "φράση",
    "αγάπη",
    "εαυτός",
    "όφελος",
    "καθοδήγηση",
    "επηρεάζω",
    "εσύ",
    "μπαμπάς",
    "άγχος",
    "ειδικός",
    "αγόρι",
    "τεστ",
    "κενός",
    "πληρωμή",
    "σούπα",
    "υποχρέωση",
    "απάντηση",
    "χαμόγελο",
    "βαθύς",
    "παράπονο",
    "προσθήκη",
    "ανασκόπηση",
    "κουτί",
    "πετσέτα",
    "ανήλικος",
    "διασκέδαση",
    "χώμα",
    "θέμα",
    "τσιγάρο",
    "διαδίκτυο",
    "κέρδος",
    "λέω",
    "είσοδος",
    "εφεδρικός",
    "περιστατικό",
    "οικογένεια",
    "αρνούμαι",
    "κλάδος",
    "δοχείο",
    "στυλό",
    "παππούς",
    "σταθερός",
    "δεξαμενή",
    "θείος",
    "κλίμα",
    "έδαφος",
    "όγκος",
    "επικοινωνία",
    "είδος",
    "ποιητής",
    "παιδί",
    "οθόνη",
    "δικό μου",
    "σταματώ",
    "γονίδιο",
    "έλλειψη",
    "φιλανθρωπία",
    "μνήμη",
    "δόντι",
    "φόβος",
    "αναφέρω",
    "μάρκετινγκ",
    "αποκαλύπτω",
    "λόγος",
    "δικαστήριο",
    "εποχή",
    "ελευθερία",
    "γη",
    "άθλημα",
    "κοινό",
    "τάξη",
    "νόμος",
    "γάντζος",
    "κερδίζω",
    "κουβαλάω",
    "μάτι",
    "μυρίζω",
    "διανομή",
    "έρευνα",
    "χώρα",
    "τολμάω",
    "ελπίδα",
    "ενώ",
    "τεντώνω",
    "βιβλιοθήκη",
    "αν",
    "καθυστερώ",
    "κολέγιο",
    "πλαστικό",
    "βιβλίο",
    "παρόν",
    "χρησιμοποιώ",
    "ανησυχώ",
    "πρωταθλητής",
    "στόχος",
    "οικονομία",
    "Μάρτιος",
    "εκλογή",
    "αντανάκλαση",
    "μεσάνυχτα",
    "ολίσθηση",
    "πληθωρισμός",
    "δράση",
    "πρόκληση",
    "κιθάρα",
    "ακτή",
    "μήλο",
    "εκστρατεία",
    "πεδίο",
    "μπουφάν",
    "αίσθηση",
    "τρόπος",
    "οπτικός",
    "αφαιρώ",
    "καιρός",
    "σκουπίδια",
    "καλώδιο",
    "μετανιώνω",
    "φίλος",
    "παραλία",
    "ιστορικός",
    "θάρρος",
    "συμπάθεια",
    "φορτηγό",
    "ένταση",
    "επιτρέπω",
    "μύτη",
    "κρεβάτι",
    "γιος",
    "πρόσωπο",
    "βάση",
    "κρέας",
    "συνήθης",
    "αέρας",
    "συνάντηση",
    "αξία",
    "παιχνίδι",
    "ανεξαρτησία",
    "φυσικός",
    "σύντομος",
    "παίζω",
    "σηκώνω",
    "πίνακας",
    "αυτή",
    "κλειδί",
    "γραφή",
    "διαλέγω",
    "εντολή",
    "πάρτι",
    "χθες",
    "άνοιξη",
    "υποψήφιος",
    "φυσική",
    "πανεπιστήμιο",
    "ανησυχία",
    "ανάπτυξη",
    "αλλαγή",
    "κορδόνι",
    "στόχος",
    "παράδειγμα",
    "δωμάτιο",
    "πικρός",
    "πουλί",
    "ποδόσφαιρο",
    "κανονικός",
    "διαχωρίζω",
    "εντύπωση",
    "ξύλο",
    "μακρύς",
    "σημασία",
    "στοκ",
    "καπέλο",
    "ηγεσία",
    "μέσα ενημέρωσης",
    "φιλοδοξία",
    "ψάρεμα",
    "δοκίμιο",
    "σαλάτα",
    "επισκευή",
    "σήμερα",
    "σχεδιαστής",
    "νύχτα",
    "τράπεζα",
    "σχέδιο",
    "αναπόφευκτος",
    "φάση",
    "τεράστιος",
    "τσιπ",
    "θυμός",
    "διακόπτης",
    "κλαίω",
    "στρίβω",
    "προσωπικότητα",
    "προσπάθεια",
    "αποθήκευση",
    "ον",
    "προετοιμασία",
    "νυχτερίδα",
    "επιλογή",
    "λευκός",
    "τεχνολογία",
    "συμβόλαιο",
    "πλευρά",
    "ενότητα",
    "σταθμός",
    "μέχρι",
    "δομή",
    "γλώσσα",
    "γεύση",
    "αλήθεια",
    "δυσκολία",
    "ομάδα",
    "όριο",
    "κύριος",
    "μετακίνηση",
    "αίσθημα",
    "φως",
    "παράδειγμα",
    "αποστολή",
    "ίσως",
    "περιμένω",
    "τροχός",
    "κατάστημα",
    "οικοδεσπότης",
    "κλασικός",
    "εναλλακτική",
    "αιτία",
    "αντιπρόσωπος",
    "συνίσταμαι",
    "τραπέζι",
    "αεροπορική εταιρεία",
    "κείμενο",
    "πισίνα",
    "χειροτεχνία",
    "εύρος",
    "καύσιμο",
    "εργαλείο",
    "συνεργάτης",
    "φορτίο",
    "είσοδος",
    "κατάθεση",
    "μισώ",
    "άρθρο",
    "βίντεο",
    "καλοκαίρι",
    "χαρακτηριστικό",
    "ακραίος",
    "κινητός",
    "νοσοκομείο",
    "πτήση",
    "φθινόπωρο",
    "σύνταξη",
    "πιάνο",
    "αποτυγχάνω",
    "αποτέλεσμα",
    "τρίβω",
    "χάσμα",
    "σύστημα",
    "αναφορά",
    "ρουφάω",
    "συνηθισμένος",
    "άνεμος",
    "νεύρο",
    "ρωτάω",
    "λάμπω",
    "σημείωση",
    "γραμμή",
    "μαμά",
    "αντίληψη",
    "αδελφός",
    "αναφορά",
    "λυγίζω",
    "χρέωση",
    "θεραπεία",
    "κόλπο",
    "όρος",
    "εργασία για το σπίτι",
    "ψήνω",
    "προσφορά",
    "κατάσταση",
    "έργο",
    "στρατηγική",
    "πορτοκαλί",
    "αφήνω",
    "ενθουσιασμός",
    "γονέας",
    "συγκεντρώνομαι",
    "συσκευή",
    "ταξιδεύω",
    "ποίηση",
    "επιχείρηση",
    "κοινωνία",
    "φιλί",
    "τέλος",
    "λαχανικό",
    "απασχολώ",
    "πρόγραμμα",
    "ώρα",
    "γενναίος",
    "εστίαση",
    "διαδικασία",
    "ταινία",
    "παράνομος",
    "γενικός",
    "καφές",
    "διαφήμιση",
    "αυτοκινητόδρομος",
    "χημεία",
    "ψυχολογία",
    "προσλαμβάνω",
    "καμπάνα",
    "συνέδριο",
    "ανακούφιση",
    "δείχνω",
    "τακτοποιώ",
    "αστείος",
    "βάρος",
    "ποιότητα",
    "κλαμπ",
    "κόρη",
    "ζώνη",
    "αγγίζω",
    "απόψε",
    "σοκ",
    "καίω",
    "δικαιολογία",
    "όνομα",
    "έρευνα",
    "τοπίο",
    "πρόοδος",
    "ικανοποίηση",
    "ψωμί",
    "καταστροφή",
    "αντικείμενο",
    "καπέλο",
    "προηγούμενος",
    "ψώνια",
    "επισκέπτομαι",
    "ανατολή",
    "φωτογραφία",
    "σπίτι",
    "ιδέα",
    "πατέρας",
    "σύγκριση",
    "γάτα",
    "σωλήνας",
    "νικητής",
    "μετράω",
    "λίμνη",
    "μάχη",
    "βραβείο",
    "ίδρυμα",
    "σκύλος",
    "κρατάω",
    "ιδανικός",
    "ανεμιστήρας",
    "αγώνας",
    "κορυφή",
    "ασφάλεια",
    "λύση",
    "κόλαση",
    "συμπέρασμα",
    "πληθυσμός",
    "στέλεχος",
    "συναγερμός",
    "μέτρηση",
    "δεύτερος",
    "τρένο",
    "φυλή",
    "οφειλόμενος",
    "ασφάλιση",
    "αφεντικό",
    "δέντρο",
    "οθόνη",
    "άρρωστος",
    "μάθημα",
    "σύρω",
    "ραντεβού",
    "φέτα",
    "ακόμα",
    "νοιάζομαι",
    "υπομονή",
    "πλούσιος",
    "αποδρώ",
    "συναίσθημα",
    "βασιλικός",
    "θηλυκός",
    "παιδική ηλικία",
    "κυβέρνηση",
    "εικόνα",
    "θέληση",
    "κάλτσα",
    "μεγάλος",
    "πύλη",
    "λάδι",
    "σταυρός",
    "καρφίτσα",
    "βελτίωση",
    "πρωτάθλημα",
    "ανόητος",
    "βοηθάω",
    "ουρανός",
    "πίσσα",
    "άνθρωπος",
    "διαμάντι",
    "περισσότερο",
    "μετάβαση",
    "δουλειά",
    "επιστήμη",
    "επιτροπή",
    "στιγμή",
    "διορθώνω",
    "διδασκαλία",
    "σκάβω",
    "ειδικός",
    "σύνθετος",
    "οδηγός",
    "άνθρωποι",
    "νεκρός",
    "φωνή",
]  # pylint: disable=line-too-long

# ISO 639-1 codes to language names.
LANGUAGE_CODES = {
    "en": "English",
    "es": "Spanish",
    "pt": "Portuguese",
    "ar": "Arabic",
    "hi": "Hindi",
    "fr": "French",
    "ru": "Russian",
    "de": "German",
    "ja": "Japanese",
    "it": "Italian",
    "bn": "Bengali",
    "uk": "Ukrainian",
    "th": "Thai",
    "ur": "Urdu",
    "ta": "Tamil",
    "te": "Telugu",
    "bg": "Bulgarian",
    "ko": "Korean",
    "pl": "Polish",
    "he": "Hebrew",
    "fa": "Persian",
    "vi": "Vietnamese",
    "ne": "Nepali",
    "sw": "Swahili",
    "kn": "Kannada",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ml": "Malayalam",
    "fi": "Finnish",
}

_ALPHABETS = "([Α-Ωα-ω])"
_PREFIXES = "(κ|Αγ|δ|Δρ)[.]"
_SUFFIXES = r"(Α\.Ε\.|Ε\.Π\.Ε\.|Ετ\.)"
_STARTERS = r"(κ\.|δ\.|Δρ\.|Καθ\.|Πλ\.|Υπολ\.|Αυτός\s|Αυτή\s|Αυτό\s|Αυτοί\s|Αυτές\s|Αυτά\s|Τους\s|Των\s|Μας\s|Εμείς\s|Αλλά\s|Ωστόσο\s|Ότι\s|Αυτό\s|Οπουδήποτε)"
_ACRONYMS = "([A-ZΑ-Ω][.][A-ZΑ-Ω][.](?:[A-ZΑ-Ω][.])?)"
_WEBSITES = "[.](com|net|org|io|gov|edu|me)"
_DIGITS = "([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"


def split_into_sentences(text):
    """Split the text into sentences.

    Args:
      text: A string that consists of more than or equal to one sentences.

    Returns:
      A list of strings where each string is a sentence.
    """
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(_PREFIXES, "\\1<prd>", text)
    text = re.sub(_WEBSITES, "<prd>\\1", text)
    text = re.sub(_DIGITS + "[.]" + _DIGITS, "\\1<prd>\\2", text)
    text = re.sub(
        _MULTIPLE_DOTS,
        lambda match: "<prd>" * len(match.group(0)) + "<stop>",
        text,
    )
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(r"\s" + _ALPHABETS + "[.] ", " \\1<prd> ", text)
    text = re.sub(_ACRONYMS + " " + _STARTERS, "\\1<stop> \\2", text)
    text = re.sub(
        _ALPHABETS + "[.]" + _ALPHABETS + "[.]" + _ALPHABETS + "[.]",
        "\\1<prd>\\2<prd>\\3<prd>",
        text,
    )
    text = re.sub(_ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + _SUFFIXES + "[.] " + _STARTERS, " \\1<stop> \\2", text)
    text = re.sub(" " + _SUFFIXES + "[.]", " \\1<prd>", text)
    text = re.sub(" " + _ALPHABETS + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


def count_words(text):
    """Counts the number of words."""
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)
    num_words = len(tokens)
    return num_words


@functools.lru_cache(maxsize=None)
def _get_sentence_tokenizer():
    return nltk.data.load("nltk:tokenizers/punkt/greek.pickle")


def count_sentences(text):
    """Count the number of sentences."""
    tokenizer = _get_sentence_tokenizer()
    tokenized_sentences = tokenizer.tokenize(text)
    return len(tokenized_sentences)


def generate_keywords(num_keywords):
    """Randomly generates a few keywords."""
    return random.sample(WORD_LIST, k=num_keywords)
